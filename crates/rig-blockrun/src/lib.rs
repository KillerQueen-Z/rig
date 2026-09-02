#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]
//! BlockRun API client and Rig integration
//!
//! [BlockRun](https://blockrun.ai) provides pay-per-request access to ~100 chat
//! models via x402 micropayments. Callers pay in USDC on Base — no API keys, no
//! accounts, no subscription. A subset of the catalogue is free and needs no
//! wallet at all.
//!
//! # Example
//! ```ignore
//! use rig_blockrun::{Client, CLAUDE_OPUS_5, GPT_56_TERRA, FREE_QWEN35_397B};
//!
//! // Paid models: sign with a funded wallet (reads BLOCKRUN_WALLET_KEY).
//! let client = Client::from_env()?;
//! let claude = client.completion_model(CLAUDE_OPUS_5);
//! let gpt = client.completion_model(GPT_56_TERRA);
//!
//! // Free models: no wallet, no payment.
//! let free = Client::free()?.completion_model(FREE_QWEN35_397B);
//! ```
//!
//! Any id the gateway serves works as a plain string, so the constants in this
//! crate are a convenience rather than a limit:
//!
//! ```ignore
//! let model = client.completion_model("zai/glm-5.3");
//! ```
//!
//! # Supported Models
//!
//! The catalogue moves faster than this crate — `GET /v1/models` is the live
//! list. As of writing it spans:
//!
//! - **Anthropic**: Claude Opus 5 / 4.8, Sonnet 5 / 4.6, Haiku 4.5, Fable 5
//! - **OpenAI**: GPT-5.6 (Sol / Terra / Luna), GPT-5.5, GPT-5.3 Codex, GPT-4o, o3, o4-mini
//! - **Google**: Gemini 3.1 Pro, 3.6 Flash, 3.5 Flash Lite, 2.5 Pro
//! - **xAI**: Grok 4.5, 4.3, 4.20, Grok Code Fast
//! - **DeepSeek**: DeepSeek V4 Pro
//! - **Moonshot**: Kimi K3, K2.7
//! - **Z.ai**: GLM-5.3, GLM-5.3 Flash
//! - **MiniMax / Qwen / Tencent / Xiaomi**, and a free NVIDIA-hosted tier
//!
//! # Payment Flow
//!
//! Paid models use the x402 protocol (version 2):
//! 1. Client makes request without payment
//! 2. Server returns 402 with payment requirements
//! 3. Client signs an EIP-712 `TransferWithAuthorization` (USDC on Base)
//! 4. Client retries with the signed `X-PAYMENT` header
//! 5. Server verifies, processes the request, settles the payment
//!
//! The private key never leaves your machine — it is only used for local
//! signing. The EIP-712 domain is pinned to the USDC contract on Base and is
//! never taken from the server's payment requirements.
//!
//! Free models skip this entirely: the gateway issues no 402, and
//! [`Client::free`] holds no key.

mod json_utils;

use async_stream::stream;
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use futures::StreamExt;
use rand::RngCore;
use reqwest::StatusCode;
use rig_core::client::CompletionClient;
use rig_core::completion::{
    self, AssistantContent, CompletionError, CompletionRequest, FinishReason,
};
use rig_core::message::{self, Document, DocumentSourceKind};
use rig_core::streaming::{self, RawStreamingChoice, RawStreamingToolCall, StreamFinal};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tracing::{Instrument, Level, enabled, info_span};

// ================================================================
// Constants
// ================================================================
const BLOCKRUN_API_BASE_URL: &str = "https://blockrun.ai/api";

// Base Mainnet
const BASE_CHAIN_ID: u64 = 8453;
const USDC_BASE: &str = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913";

/// Service code attached to every payment under the `builder-code` extension,
/// matching the JS and Python SDKs. It is how BlockRun attributes settled
/// payments back to the SDK that produced them.
const BLOCKRUN_SERVICE_CODE: &str = "blockrun";

// ================================================================
// Available Models
// ================================================================
//
// A curated selection of the catalogue. Any model id the gateway serves works
// as a plain `&str` — `client.completion_model("provider/model")` — so this
// list is a convenience, not a limit. The live catalogue is `GET /v1/models`.

// ---------- Free tier — no wallet, no payment ----------
// Free models bypass the x402 flow entirely: the gateway never issues a 402
// for them, so [`Client::free`] (which holds no private key) can serve these.
// They are rate limited per IP rather than per payment.

pub const FREE_QWEN35_397B: &str = "nvidia/qwen3.5-397b-a17b";
pub const FREE_NEMOTRON_3_SUPER_120B: &str = "nvidia/nemotron-3-super-120b";
pub const FREE_MISTRAL_LARGE_3_675B: &str = "nvidia/mistral-large-3-675b";
pub const FREE_LLAMA_4_MAVERICK: &str = "nvidia/llama-4-maverick";
pub const FREE_GPT_OSS_20B: &str = "nvidia/gpt-oss-20b";
pub const FREE_NEMOTRON_NANO_9B: &str = "nvidia/nemotron-nano-9b-v2";

// ---------- Anthropic ----------
pub const CLAUDE_OPUS_5: &str = "anthropic/claude-opus-5";
pub const CLAUDE_OPUS_48: &str = "anthropic/claude-opus-4.8";
pub const CLAUDE_SONNET_5: &str = "anthropic/claude-sonnet-5";
pub const CLAUDE_SONNET_46: &str = "anthropic/claude-sonnet-4.6";
pub const CLAUDE_HAIKU_45: &str = "anthropic/claude-haiku-4.5";
pub const CLAUDE_FABLE_5: &str = "anthropic/claude-fable-5";

// ---------- OpenAI ----------
pub const GPT_56_SOL: &str = "openai/gpt-5.6-sol";
pub const GPT_56_TERRA: &str = "openai/gpt-5.6-terra";
pub const GPT_56_LUNA: &str = "openai/gpt-5.6-luna";
pub const GPT_55: &str = "openai/gpt-5.5";
pub const GPT_54_MINI: &str = "openai/gpt-5.4-mini";
pub const GPT_53_CODEX: &str = "openai/gpt-5.3-codex";
pub const GPT_4O: &str = "openai/gpt-4o";
pub const GPT_4O_MINI: &str = "openai/gpt-4o-mini";
pub const GPT_O3: &str = "openai/o3";
pub const GPT_O4_MINI: &str = "openai/o4-mini";

// ---------- Google ----------
pub const GEMINI_31_PRO: &str = "google/gemini-3.1-pro";
pub const GEMINI_36_FLASH: &str = "google/gemini-3.6-flash";
pub const GEMINI_35_FLASH_LITE: &str = "google/gemini-3.5-flash-lite";
pub const GEMINI_25_PRO: &str = "google/gemini-2.5-pro";

// ---------- xAI ----------
pub const GROK_45: &str = "xai/grok-4.5";
pub const GROK_43: &str = "xai/grok-4.3";
pub const GROK_420_REASONING: &str = "xai/grok-4.20-reasoning";
pub const GROK_41_FAST_REASONING: &str = "xai/grok-4-1-fast-reasoning";
pub const GROK_CODE_FAST_1: &str = "xai/grok-code-fast-1";

// ---------- DeepSeek ----------
pub const DEEPSEEK_V4_PRO: &str = "deepseek/deepseek-v4-pro";

// ---------- Moonshot ----------
pub const KIMI_K3: &str = "moonshot/kimi-k3";
pub const KIMI_K27: &str = "moonshot/kimi-k2.7";

// ---------- Z.ai ----------
pub const GLM_53: &str = "zai/glm-5.3";
pub const GLM_53_FLASH: &str = "zai/glm-5.3-flash";

// ---------- MiniMax / Qwen ----------
pub const MINIMAX_M3: &str = "minimax/minimax-m3";
pub const QWEN37_MAX: &str = "qwen/qwen3.7-max";
pub const QWEN37_FLASH: &str = "qwen/qwen3.7-flash";

// ================================================================
// x402 Payment Types
// ================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PaymentAccept {
    scheme: String,
    network: String,
    amount: String,
    asset: String,
    #[serde(rename = "payTo")]
    pay_to: String,
    #[serde(rename = "maxTimeoutSeconds")]
    max_timeout_seconds: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    extra: Option<PaymentExtra>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PaymentExtra {
    name: String,
    version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PaymentRequired {
    #[serde(rename = "x402Version")]
    x402_version: u32,
    accepts: Vec<PaymentAccept>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resource: Option<ResourceInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResourceInfo {
    url: String,
    description: String,
    #[serde(rename = "mimeType")]
    mime_type: String,
}

#[derive(Debug, Serialize)]
struct PaymentPayload {
    #[serde(rename = "x402Version")]
    x402_version: u32,
    resource: ResourceInfo,
    accepted: PaymentAccepted,
    payload: SignaturePayload,
    #[serde(skip_serializing_if = "Option::is_none")]
    extensions: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct PaymentAccepted {
    scheme: String,
    network: String,
    amount: String,
    asset: String,
    #[serde(rename = "payTo")]
    pay_to: String,
    #[serde(rename = "maxTimeoutSeconds")]
    max_timeout_seconds: u64,
    extra: PaymentExtra,
}

#[derive(Debug, Serialize)]
struct SignaturePayload {
    signature: String,
    authorization: Authorization,
}

#[derive(Debug, Serialize)]
struct Authorization {
    from: String,
    to: String,
    value: String,
    #[serde(rename = "validAfter")]
    valid_after: String,
    #[serde(rename = "validBefore")]
    valid_before: String,
    nonce: String,
}

// ================================================================
// EIP-712 Signing
// ================================================================

/// Left-pad a `0x`-prefixed 20-byte address into an EIP-712 word.
///
/// The address is server-controlled (`payTo` comes straight off the 402), so a
/// malformed value has to be an error rather than a panic — the previous
/// `hex::decode(&addr[2..]).expect(..)` plus fixed-width `copy_from_slice`
/// aborted the caller's process on any short, long, or non-hex input.
fn address_to_word(label: &str, address: &str) -> Result<[u8; 32], CompletionError> {
    let body = address.strip_prefix("0x").unwrap_or(address);

    let bytes = hex::decode(body).map_err(|e| {
        CompletionError::ProviderError(format!("{label} is not valid hex ({address}): {e}"))
    })?;

    let bytes: [u8; 20] = bytes.try_into().map_err(|_| {
        CompletionError::ProviderError(format!("{label} is not a 20-byte address: {address}"))
    })?;

    let mut word = [0u8; 32];
    word[12..].copy_from_slice(&bytes);
    Ok(word)
}

/// EIP-712 domain separator for USDC on Base.
///
/// Pinned to the USDC contract, never derived from the server's `extra` — a
/// facilitator that could choose the domain could have a payment signed for a
/// different token or chain.
fn eip712_domain_separator() -> Result<[u8; 32], CompletionError> {
    use sha3::{Digest, Keccak256};

    let type_hash = Keccak256::digest(
        b"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)",
    );

    let name_hash = Keccak256::digest(b"USD Coin");
    let version_hash = Keccak256::digest(b"2");

    let mut chain_id_bytes = [0u8; 32];
    chain_id_bytes[24..].copy_from_slice(&BASE_CHAIN_ID.to_be_bytes());

    let contract_padded = address_to_word("USDC contract address", USDC_BASE)?;

    let mut data = Vec::with_capacity(160);
    data.extend_from_slice(&type_hash);
    data.extend_from_slice(&name_hash);
    data.extend_from_slice(&version_hash);
    data.extend_from_slice(&chain_id_bytes);
    data.extend_from_slice(&contract_padded);

    Ok(Keccak256::digest(&data).into())
}

/// Hash for TransferWithAuthorization struct
fn transfer_struct_hash(
    from: &str,
    to: &str,
    value: &str,
    valid_after: u64,
    valid_before: u64,
    nonce: &[u8; 32],
) -> Result<[u8; 32], CompletionError> {
    use sha3::{Digest, Keccak256};

    let type_hash = Keccak256::digest(
        b"TransferWithAuthorization(address from,address to,uint256 value,uint256 validAfter,uint256 validBefore,bytes32 nonce)",
    );

    let from_padded = address_to_word("payer address", from)?;
    let to_padded = address_to_word("payTo address", to)?;

    let value_num: u128 = value.parse().map_err(|e| {
        CompletionError::ProviderError(format!("payment amount is not a number ({value}): {e}"))
    })?;
    let mut value_bytes = [0u8; 32];
    value_bytes[16..].copy_from_slice(&value_num.to_be_bytes());

    let mut valid_after_bytes = [0u8; 32];
    valid_after_bytes[24..].copy_from_slice(&valid_after.to_be_bytes());

    let mut valid_before_bytes = [0u8; 32];
    valid_before_bytes[24..].copy_from_slice(&valid_before.to_be_bytes());

    let mut data = Vec::with_capacity(224);
    data.extend_from_slice(&type_hash);
    data.extend_from_slice(&from_padded);
    data.extend_from_slice(&to_padded);
    data.extend_from_slice(&value_bytes);
    data.extend_from_slice(&valid_after_bytes);
    data.extend_from_slice(&valid_before_bytes);
    data.extend_from_slice(nonce);

    Ok(Keccak256::digest(&data).into())
}

/// Create EIP-712 typed data hash
fn eip712_hash(struct_hash: [u8; 32]) -> Result<[u8; 32], CompletionError> {
    use sha3::{Digest, Keccak256};

    let domain_separator = eip712_domain_separator()?;

    let mut data = Vec::with_capacity(66);
    data.extend_from_slice(&[0x19, 0x01]);
    data.extend_from_slice(&domain_separator);
    data.extend_from_slice(&struct_hash);

    Ok(Keccak256::digest(&data).into())
}

/// Sign EIP-712 typed data with secp256k1 private key
fn sign_eip712(private_key: &[u8; 32], message_hash: [u8; 32]) -> Result<String, CompletionError> {
    use k256::ecdsa::{RecoveryId, Signature, SigningKey};

    let signing_key = SigningKey::from_bytes(private_key.into())
        .map_err(|e| CompletionError::ProviderError(format!("Invalid private key: {}", e)))?;

    let (signature, recovery_id): (Signature, RecoveryId) = signing_key
        .sign_prehash_recoverable(&message_hash)
        .map_err(|e| CompletionError::ProviderError(format!("Signing failed: {}", e)))?;

    let mut sig_bytes = [0u8; 65];
    sig_bytes[..64].copy_from_slice(&signature.to_bytes());
    sig_bytes[64] = recovery_id.to_byte() + 27;

    Ok(format!("0x{}", hex::encode(sig_bytes)))
}

/// Get wallet address from private key
fn get_address_from_private_key(private_key: &[u8; 32]) -> Result<String, CompletionError> {
    use k256::ecdsa::SigningKey;
    use sha3::{Digest, Keccak256};

    let signing_key = SigningKey::from_bytes(private_key.into())
        .map_err(|e| CompletionError::ProviderError(format!("Invalid private key: {}", e)))?;

    let public_key = signing_key.verifying_key();
    let public_key_bytes = public_key.to_encoded_point(false);
    // Drop the SEC1 `0x04` uncompressed-point tag before hashing.
    let public_key_uncompressed = public_key_bytes.as_bytes().get(1..).ok_or_else(|| {
        CompletionError::ProviderError("public key encoding was empty".to_string())
    })?;

    let hash = Keccak256::digest(public_key_uncompressed);
    // An Ethereum address is the last 20 bytes of the keccak hash.
    let address_bytes = hash.get(12..).ok_or_else(|| {
        CompletionError::ProviderError("keccak digest was too short for an address".to_string())
    })?;

    Ok(format!("0x{}", hex::encode(address_bytes)))
}

// ================================================================
// BlockRun Auth
// ================================================================

#[derive(Clone)]
struct BlockRunAuth {
    private_key: [u8; 32],
    address: String,
}

impl std::fmt::Debug for BlockRunAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockRunAuth")
            .field("address", &self.address)
            .field("private_key", &"[REDACTED]")
            .finish()
    }
}

impl BlockRunAuth {
    fn new(private_key_hex: &str) -> Result<Self, CompletionError> {
        let key_hex = private_key_hex
            .strip_prefix("0x")
            .unwrap_or(private_key_hex);
        let key_bytes = hex::decode(key_hex)
            .map_err(|e| CompletionError::ProviderError(format!("Invalid hex key: {}", e)))?;

        if key_bytes.len() != 32 {
            return Err(CompletionError::ProviderError(
                "Private key must be 32 bytes".to_string(),
            ));
        }

        let mut private_key = [0u8; 32];
        private_key.copy_from_slice(&key_bytes);

        let address = get_address_from_private_key(&private_key)?;

        Ok(Self {
            private_key,
            address,
        })
    }

    fn address(&self) -> &str {
        &self.address
    }

    /// Create a signed x402 payment payload
    fn create_payment(
        &self,
        payment_required: &PaymentRequired,
    ) -> Result<String, CompletionError> {
        let accept = payment_required
            .accepts
            .first()
            .ok_or_else(|| CompletionError::ProviderError("No payment options".to_string()))?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| CompletionError::ProviderError(format!("Time error: {}", e)))?
            .as_secs();

        let valid_after = now.saturating_sub(600);
        let valid_before = now + accept.max_timeout_seconds;

        let mut nonce = [0u8; 32];
        rand::rng().fill_bytes(&mut nonce);
        let nonce_hex = format!("0x{}", hex::encode(nonce));

        let struct_hash = transfer_struct_hash(
            &self.address,
            &accept.pay_to,
            &accept.amount,
            valid_after,
            valid_before,
            &nonce,
        )?;

        let message_hash = eip712_hash(struct_hash)?;
        let signature = sign_eip712(&self.private_key, message_hash)?;

        let payload = PaymentPayload {
            x402_version: 2,
            resource: payment_required
                .resource
                .clone()
                .unwrap_or_else(|| ResourceInfo {
                    url: "https://blockrun.ai/api/v1/chat/completions".to_string(),
                    description: "BlockRun AI API call".to_string(),
                    mime_type: "application/json".to_string(),
                }),
            accepted: PaymentAccepted {
                scheme: accept.scheme.clone(),
                network: accept.network.clone(),
                amount: accept.amount.clone(),
                asset: accept.asset.clone(),
                pay_to: accept.pay_to.clone(),
                max_timeout_seconds: accept.max_timeout_seconds,
                extra: PaymentExtra {
                    name: "USD Coin".to_string(),
                    version: "2".to_string(),
                },
            },
            payload: SignaturePayload {
                signature,
                authorization: Authorization {
                    from: self.address.clone(),
                    to: accept.pay_to.clone(),
                    value: accept.amount.clone(),
                    valid_after: valid_after.to_string(),
                    valid_before: valid_before.to_string(),
                    nonce: nonce_hex,
                },
            },
            // Attribute the settled payment back to this SDK, matching the
            // `builder-code` extension the JS and Python SDKs send.
            extensions: Some(serde_json::json!({
                "builder-code": { "info": { "s": [BLOCKRUN_SERVICE_CODE] } }
            })),
        };

        let json = serde_json::to_string(&payload)
            .map_err(|e| CompletionError::ProviderError(format!("JSON error: {}", e)))?;

        Ok(BASE64.encode(json.as_bytes()))
    }
}

// ================================================================
// BlockRun Client
// ================================================================

/// The HTTP transport both constructors share.
///
/// The 120s timeout covers BlockRun's own free-tier cascade budget, which can
/// legitimately spend ~110s before returning.
fn build_http_client() -> Result<reqwest::Client, CompletionError> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(|e| CompletionError::ProviderError(format!("HTTP client error: {e}")))
}

/// BlockRun API client for x402 micropayment-based AI inference
#[derive(Clone)]
pub struct Client {
    /// `None` for a wallet-less client. Free models never trigger the x402
    /// flow, so they need no key; a paid model on such a client fails at the
    /// 402 with [`CompletionError::ProviderError`] rather than panicking.
    auth: Option<BlockRunAuth>,
    http_client: reqwest::Client,
    #[allow(dead_code)] // Reserved for custom endpoint support
    base_url: String,
}

impl Client {
    /// Create a new BlockRun client with the given wallet private key.
    ///
    /// BlockRun uses x402 micropayments instead of API keys.
    /// The private key is used to sign payment authorizations locally.
    /// It never leaves your machine - only the signature is sent.
    ///
    /// # Example
    /// ```ignore
    /// let client = Client::from_private_key("0x...")?;
    /// ```
    pub fn from_private_key(private_key: &str) -> Result<Self, CompletionError> {
        let auth = BlockRunAuth::new(private_key)?;

        Ok(Self {
            auth: Some(auth),
            http_client: build_http_client()?,
            base_url: BLOCKRUN_API_BASE_URL.to_string(),
        })
    }

    /// Create a client from the `BLOCKRUN_WALLET_KEY` environment variable.
    ///
    /// Returns an error rather than panicking when the variable is missing or
    /// holds a key that does not parse — a missing environment variable is an
    /// ordinary misconfiguration, not a bug in the caller.
    pub fn from_env() -> Result<Self, CompletionError> {
        let private_key = std::env::var("BLOCKRUN_WALLET_KEY").map_err(|_| {
            CompletionError::ProviderError(
                "BLOCKRUN_WALLET_KEY is not set. Set it to a hex private key, or use \
                 `Client::free()` for the free models."
                    .to_string(),
            )
        })?;
        Self::from_private_key(&private_key)
    }

    /// Create a client with no wallet, for the free tier.
    ///
    /// Free models (the `FREE_*` constants) bypass x402 entirely — the gateway
    /// never issues a 402 for them — so no key is needed. They are rate limited
    /// per IP instead. Asking such a client for a paid model returns a
    /// [`CompletionError::ProviderError`] when the 402 arrives.
    ///
    /// # Example
    /// ```ignore
    /// let client = Client::free()?;
    /// let model = client.completion_model(rig_blockrun::FREE_QWEN35_397B);
    /// ```
    pub fn free() -> Result<Self, CompletionError> {
        Ok(Self {
            auth: None,
            http_client: build_http_client()?,
            base_url: BLOCKRUN_API_BASE_URL.to_string(),
        })
    }

    /// The wallet address this client signs payments with, or `None` for a
    /// free-tier client built by [`Client::free`].
    pub fn address(&self) -> Option<&str> {
        self.auth.as_ref().map(|auth| auth.address())
    }

    /// The signer, or the error a paid request should fail with when the client
    /// has no wallet.
    fn signer(&self) -> Result<&BlockRunAuth, CompletionError> {
        self.auth.as_ref().ok_or_else(|| {
            CompletionError::ProviderError(
                "this model requires payment, but the client was built with `Client::free()` \
                 (no wallet). Use `Client::from_env()` or `Client::from_private_key()`, or pick \
                 one of the free models."
                    .to_string(),
            )
        })
    }

    #[allow(dead_code)] // Reserved for future API extensions
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));
        self.http_client
            .post(url)
            .header("Content-Type", "application/json")
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;

    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel {
        CompletionModel::new(self.clone(), &model.into())
    }
}

// ================================================================
// API Response Types
// ================================================================

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Provider response id (`chatcmpl-…`). Response-scoped, so it lands in
    /// `CompletionResponse::response_id` rather than `message_id`.
    #[serde(default)]
    pub id: Option<String>,
    /// The model the gateway actually served. Worth surfacing: BlockRun
    /// transparently falls back to another model when a free primary is
    /// saturated, so this is not always the model that was requested.
    #[serde(default)]
    pub model: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Usage {
    pub completion_tokens: u32,
    pub prompt_tokens: u32,
    pub total_tokens: u32,
    /// Cache accounting, when the upstream reports it.
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// Reasoning accounting, when the upstream reports it. Most of BlockRun's
    /// catalogue is reasoning-capable now, so this is the common case rather
    /// than the exception.
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: u32,
}

impl Usage {
    fn new() -> Self {
        Self::default()
    }

    /// Project the wire's token counts onto rig's `Usage`.
    ///
    /// rig 0.42 widened `Usage` with cache and reasoning counters. Anything
    /// the gateway does not report stays at zero, which is the documented
    /// "not reported" sentinel rather than a claim of zero usage.
    fn to_rig_usage(&self) -> completion::Usage {
        completion::Usage {
            input_tokens: self.prompt_tokens as u64,
            output_tokens: self.completion_tokens as u64,
            total_tokens: self.total_tokens as u64,
            cached_input_tokens: self
                .prompt_tokens_details
                .as_ref()
                .map(|details| details.cached_tokens as u64)
                .unwrap_or(0),
            reasoning_tokens: self
                .completion_tokens_details
                .as_ref()
                .map(|details| details.reasoning_tokens as u64)
                .unwrap_or(0),
            ..completion::Usage::new()
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    System {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(
            default,
            deserialize_with = "json_utils::null_or_vec",
            skip_serializing_if = "Vec::is_empty"
        )]
        tool_calls: Vec<ToolCall>,
    },
    #[serde(rename = "tool")]
    ToolResult {
        tool_call_id: String,
        content: String,
    },
}

impl Message {
    pub fn system(content: &str) -> Self {
        Message::System {
            content: content.to_owned(),
            name: None,
        }
    }
}

fn tool_result_to_message(tool_result: message::ToolResult) -> Message {
    // 0.42 split the identifier in two: `call` is rig's correlation handle
    // (always present, sometimes minted), `provider` is what the provider
    // itself issued. Only the latter may travel back on the wire, so prefer it
    // and fall back to rig's handle for wires that issued none.
    // `ProviderCallId` also carries an output-item id for dual-identifier
    // wires; chat-completions only echoes `call_id`.
    let tool_call_id = tool_result
        .provider
        .as_ref()
        .map(|provider| provider.call_id.clone())
        .unwrap_or_else(|| tool_result.call.to_string());

    let content = match tool_result.content.first() {
        Some(message::ToolResultContent::Text(text)) => text.text.clone(),
        Some(message::ToolResultContent::Image(_)) => String::from("[Image]"),
        // New in 0.42: a tool runtime can hand back structured JSON rather
        // than pre-rendered text. Serialize it instead of dropping it.
        Some(message::ToolResultContent::Json { value }) => value.to_string(),
        None => String::new(),
    };

    Message::ToolResult {
        tool_call_id,
        content,
    }
}

fn tool_call_to_api(tool_call: message::ToolCall) -> ToolCall {
    let id = tool_call
        .provider
        .as_ref()
        .map(|provider| provider.call_id.clone())
        .unwrap_or_else(|| tool_call.id.to_string());

    ToolCall {
        id,
        index: 0,
        r#type: ToolType::Function,
        function: Function {
            name: tool_call.function.name,
            arguments: tool_call.function.arguments,
        },
    }
}

fn message_to_api(msg: message::Message) -> Result<Vec<Message>, message::MessageError> {
    match msg {
        // rig 0.42 removed `CompletionRequest::preamble`; system instructions
        // now arrive as an ordinary history message and pass straight through.
        message::Message::System { content } => Ok(vec![Message::system(&content)]),
        message::Message::User { content } => {
            let mut messages = vec![];

            let tool_results: Vec<Message> = content
                .clone()
                .into_iter()
                .filter_map(|c| match c {
                    message::UserContent::ToolResult(tool_result) => {
                        Some(tool_result_to_message(tool_result))
                    }
                    _ => None,
                })
                .collect();

            messages.extend(tool_results);

            let text_messages: Vec<Message> =
                content
                    .into_iter()
                    .filter_map(|c| match c {
                        message::UserContent::Text(text) => Some(Message::User {
                            content: text.text,
                            name: None,
                        }),
                        message::UserContent::Document(Document {
                            data:
                                DocumentSourceKind::Base64(content)
                                | DocumentSourceKind::String(content),
                            ..
                        }) => Some(Message::User {
                            content,
                            name: None,
                        }),
                        _ => None,
                    })
                    .collect();
            messages.extend(text_messages);

            Ok(messages)
        }
        message::Message::Assistant { content, .. } => {
            let mut messages: Vec<Message> = vec![];
            let mut text_content = String::new();

            content.iter().for_each(|c| {
                if let message::AssistantContent::Text(text) = c {
                    text_content.push_str(text.text());
                }
            });

            messages.push(Message::Assistant {
                content: text_content,
                name: None,
                tool_calls: vec![],
            });

            let tool_calls: Vec<ToolCall> = content
                .clone()
                .into_iter()
                .filter_map(|c| match c {
                    message::AssistantContent::ToolCall(tool_call) => {
                        Some(tool_call_to_api(tool_call))
                    }
                    _ => None,
                })
                .collect();

            if !tool_calls.is_empty() {
                messages.push(Message::Assistant {
                    content: "".to_string(),
                    name: None,
                    tool_calls,
                });
            }

            Ok(messages)
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
    pub id: String,
    /// Position within the turn's tool calls.
    ///
    /// Only streaming deltas carry this — a non-streaming
    /// `/v1/chat/completions` body omits it entirely, so requiring it made
    /// every unary tool-calling response fail to deserialize.
    #[serde(default)]
    pub index: usize,
    #[serde(default)]
    pub r#type: ToolType,
    pub function: Function,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    #[serde(with = "json_utils::stringified_json")]
    pub arguments: serde_json::Value,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: completion::ToolDefinition,
}

impl From<completion::ToolDefinition> for ToolDefinition {
    fn from(tool: completion::ToolDefinition) -> Self {
        Self {
            r#type: "function".into(),
            function: tool,
        }
    }
}

/// Normalize a BlockRun wire response into rig's provider-agnostic response.
///
/// rig 0.42 dropped the `CompletionResponse<T>` raw-response generic; the
/// provider's own body now travels in the `raw` field instead, so nothing is
/// lost — `CompletionResponse::deserialize(&resp.raw)` recovers this type.
impl TryFrom<CompletionResponse> for completion::CompletionResponse {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        let finish_reason = choice.finish_reason.as_deref().map(parse_finish_reason);

        let content = match &choice.message {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut content = if content.trim().is_empty() {
                    vec![]
                } else {
                    vec![AssistantContent::text(content)]
                };

                content.extend(tool_calls.iter().map(|call| {
                    AssistantContent::tool_call(
                        &call.id,
                        &call.function.name,
                        call.function.arguments.clone(),
                    )
                }));
                Ok(content)
            }
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a valid message or tool call".into(),
            )),
        }?;

        if content.is_empty() {
            return Err(CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            ));
        }

        let usage = response.usage.to_rig_usage();

        let response_id = response.id.clone();
        let model = response.model.clone();
        let raw = serde_json::to_value(&response)?;

        Ok(
            completion::CompletionResponse::new(content, usage, "blockrun")
                .with_optional_response_id(response_id)
                .with_optional_model(model)
                .with_optional_finish_reason(finish_reason)
                .with_raw(raw),
        )
    }
}

/// Map an OpenAI-style `finish_reason` onto rig's normalized vocabulary.
///
/// Anything outside the known set is carried through verbatim rather than
/// dropped — BlockRun fronts many upstreams and they do not all agree on the
/// spelling.
fn parse_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" | "end_turn" | "stop_sequence" => FinishReason::Stop,
        "length" | "max_tokens" => FinishReason::Length,
        "tool_calls" | "function_call" | "tool_use" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_string()),
    }
}

// ================================================================
// Completion Request
// ================================================================

#[derive(Debug, Serialize, Deserialize)]
struct BlockRunCompletionRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    additional_params: Option<serde_json::Value>,
}

/// The system instructions carried by a request, for telemetry.
///
/// rig 0.42 folded `CompletionRequest::preamble` into `chat_history` as
/// [`message::Message::System`], so the `gen_ai.system_instructions` span field
/// has to read them back out. Multiple system turns are joined in order.
fn system_instructions(req: &CompletionRequest) -> Option<String> {
    let joined = req
        .chat_history
        .iter()
        .filter_map(|msg| match msg {
            message::Message::System { content } => Some(content.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");

    (!joined.is_empty()).then_some(joined)
}

fn build_request(
    model: &str,
    req: CompletionRequest,
) -> Result<BlockRunCompletionRequest, CompletionError> {
    // Documents are prepended so they precede the turn that refers to them,
    // which is where the old `preamble`-first ordering put them too.
    let mut full_history: Vec<Message> = Vec::new();

    if let Some(docs) = req.normalized_documents() {
        full_history.extend(message_to_api(docs)?);
    }

    let chat_history: Vec<Message> = req
        .chat_history
        .clone()
        .into_iter()
        .map(message_to_api)
        .collect::<Result<Vec<Vec<Message>>, _>>()?
        .into_iter()
        .flatten()
        .collect();

    full_history.extend(chat_history);

    Ok(BlockRunCompletionRequest {
        model: model.to_string(),
        messages: full_history,
        temperature: req.temperature,
        tools: req
            .tools
            .clone()
            .into_iter()
            .map(ToolDefinition::from)
            .collect::<Vec<_>>(),
        tool_choice: None,
        additional_params: req.additional_params,
    })
}

// ================================================================
// Completion Model
// ================================================================

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    model: String,
}

impl CompletionModel {
    fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl completion::CompletionModel for CompletionModel {
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "blockrun",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        span.record(
            "gen_ai.system_instructions",
            system_instructions(&completion_request),
        );

        let request = build_request(&self.model, completion_request)?;

        if enabled!(Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "BlockRun completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let url = format!("{}/v1/chat/completions", BLOCKRUN_API_BASE_URL);

        let client = self.client.clone();

        async move {
            // First request - will return 402 with payment requirements
            let initial_response = client
                .http_client
                .post(&url)
                .header("Content-Type", "application/json")
                .body(body.clone())
                .send()
                .await
                .map_err(|e| CompletionError::ProviderError(format!("Request failed: {}", e)))?;

            let status = initial_response.status();

            // Handle 402 Payment Required
            if status == StatusCode::PAYMENT_REQUIRED {
                let payment_header = initial_response
                    .headers()
                    .get("x-payment-required")
                    .or_else(|| initial_response.headers().get("payment-required"))
                    .ok_or_else(|| {
                        CompletionError::ProviderError(
                            "402 response missing payment header".to_string(),
                        )
                    })?
                    .to_str()
                    .map_err(|_| {
                        CompletionError::ProviderError("Invalid payment header".to_string())
                    })?
                    .to_string();

                let payment_required_json = BASE64.decode(&payment_header).map_err(|e| {
                    CompletionError::ProviderError(format!(
                        "Failed to decode payment header: {}",
                        e
                    ))
                })?;

                let payment_required: PaymentRequired =
                    serde_json::from_slice(&payment_required_json).map_err(|e| {
                        CompletionError::ProviderError(format!(
                            "Failed to parse payment requirements: {}",
                            e
                        ))
                    })?;

                let payment_payload = client.signer()?.create_payment(&payment_required)?;

                let paid_response = client
                    .http_client
                    .post(&url)
                    .header("Content-Type", "application/json")
                    .header("payment", &payment_payload)
                    .header("x-payment", &payment_payload)
                    .body(body)
                    .send()
                    .await
                    .map_err(|e| {
                        CompletionError::ProviderError(format!("Paid request failed: {}", e))
                    })?;

                let paid_status = paid_response.status();
                let response_body = paid_response.bytes().await.map_err(|e| {
                    CompletionError::ProviderError(format!("Failed to read response: {}", e))
                })?;

                if paid_status.is_success() {
                    match serde_json::from_slice::<ApiResponse<CompletionResponse>>(&response_body)?
                    {
                        ApiResponse::Ok(response) => {
                            let span = tracing::Span::current();
                            span.record("gen_ai.usage.input_tokens", response.usage.prompt_tokens);
                            span.record(
                                "gen_ai.usage.output_tokens",
                                response.usage.completion_tokens,
                            );
                            if enabled!(Level::TRACE) {
                                tracing::trace!(target: "rig::completions",
                                    "BlockRun completion response: {}",
                                    serde_json::to_string_pretty(&response)?
                                );
                            }
                            response.try_into()
                        }
                        ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
                    }
                } else {
                    Err(CompletionError::ProviderError(
                        String::from_utf8_lossy(&response_body).to_string(),
                    ))
                }
            } else if status.is_success() {
                let response_body = initial_response.bytes().await.map_err(|e| {
                    CompletionError::ProviderError(format!("Failed to read response: {}", e))
                })?;
                match serde_json::from_slice::<ApiResponse<CompletionResponse>>(&response_body)? {
                    ApiResponse::Ok(response) => response.try_into(),
                    ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
                }
            } else {
                let response_body = initial_response.bytes().await.map_err(|e| {
                    CompletionError::ProviderError(format!("Failed to read response: {}", e))
                })?;
                Err(CompletionError::ProviderError(
                    String::from_utf8_lossy(&response_body).to_string(),
                ))
            }
        }
        .instrument(span)
        .await
    }

    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse, CompletionError> {
        let preamble = system_instructions(&completion_request);
        let mut request = build_request(&self.model, completion_request)?;

        let params = json_utils::merge(
            request.additional_params.unwrap_or(serde_json::json!({})),
            serde_json::json!({"stream": true, "stream_options": {"include_usage": true}}),
        );
        request.additional_params = Some(params);

        if enabled!(Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "BlockRun streaming completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let url = format!("{}/v1/chat/completions", BLOCKRUN_API_BASE_URL);

        // First request - will return 402 with payment requirements
        let initial_response = self
            .client
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .body(body.clone())
            .send()
            .await
            .map_err(|e| CompletionError::ProviderError(format!("Request failed: {}", e)))?;

        let status = initial_response.status();

        // Handle 402 Payment Required
        let payment_payload = if status == StatusCode::PAYMENT_REQUIRED {
            let payment_header = initial_response
                .headers()
                .get("x-payment-required")
                .or_else(|| initial_response.headers().get("payment-required"))
                .ok_or_else(|| {
                    CompletionError::ProviderError(
                        "402 response missing payment header".to_string(),
                    )
                })?
                .to_str()
                .map_err(|_| CompletionError::ProviderError("Invalid payment header".to_string()))?
                .to_string();

            let payment_required_json = BASE64.decode(&payment_header).map_err(|e| {
                CompletionError::ProviderError(format!("Failed to decode payment header: {}", e))
            })?;

            let payment_required: PaymentRequired = serde_json::from_slice(&payment_required_json)
                .map_err(|e| {
                    CompletionError::ProviderError(format!(
                        "Failed to parse payment requirements: {}",
                        e
                    ))
                })?;

            self.client.signer()?.create_payment(&payment_required)?
        } else {
            return Err(CompletionError::ProviderError(
                "Expected 402 response for payment".to_string(),
            ));
        };

        // Make the paid streaming request
        let response = self
            .client
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("payment", &payment_payload)
            .header("x-payment", &payment_payload)
            .body(body)
            .send()
            .await
            .map_err(|e| CompletionError::ProviderError(format!("Paid request failed: {}", e)))?;

        if !response.status().is_success() {
            let body = response.bytes().await.map_err(|e| {
                CompletionError::ProviderError(format!("Failed to read error response: {}", e))
            })?;
            return Err(CompletionError::ProviderError(
                String::from_utf8_lossy(&body).to_string(),
            ));
        }

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "blockrun",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let _guard = span.enter();

        let stream = stream! {
            let mut final_usage = Usage::new();
            let mut final_finish_reason: Option<String> = None;
            let mut final_response_id: Option<String> = None;
            let mut final_model: Option<String> = None;
            let mut calls: HashMap<usize, (String, String, String)> = HashMap::new();
            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::error!("Stream error: {:?}", e);
                        yield Err(CompletionError::ResponseError(e.to_string()));
                        break;
                    }
                };

                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete SSE events
                while let Some(pos) = buffer.find("\n\n") {
                    let event = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    for line in event.lines() {
                        if let Some(data) = line.strip_prefix("data: ") {
                            if data.trim().is_empty() || data == "[DONE]" {
                                continue;
                            }

                            let chunk_data = match serde_json::from_str::<StreamingCompletionChunk>(data) {
                                Ok(chunk_data) => chunk_data,
                                Err(err) => {
                                    tracing::debug!("Couldn't parse SSE payload: {err:?}");
                                    continue;
                                }
                            };

                            if let Some(choice) = chunk_data.choices.first() {
                                let delta = &choice.delta;

                                for tool_call in &delta.tool_calls {
                                    let function = &tool_call.function;

                                    // A fragment either opens a call (it names the
                                    // function) or extends the open one's arguments.
                                    match function.name.as_deref().filter(|name| !name.is_empty()) {
                                        Some(name) => {
                                            let id = tool_call.id.clone().unwrap_or_default();
                                            calls.insert(
                                                tool_call.index,
                                                (id, name.to_string(), String::new()),
                                            );
                                        }
                                        None => {
                                            if let Some(arguments) = &function.arguments
                                                && let Some((id, name, existing_args)) =
                                                    calls.get(&tool_call.index)
                                            {
                                                let combined = format!("{existing_args}{arguments}");
                                                calls.insert(
                                                    tool_call.index,
                                                    (id.clone(), name.clone(), combined),
                                                );
                                            }
                                        }
                                    }
                                }

                                if let Some(content) = &delta.content {
                                    yield Ok(RawStreamingChoice::Message(content.clone()));
                                }
                            }

                            if let Some(usage) = chunk_data.usage {
                                final_usage = usage;
                            }
                            if final_response_id.is_none() {
                                final_response_id = chunk_data.id.clone();
                            }
                            if final_model.is_none() {
                                final_model = chunk_data.model.clone();
                            }
                            if let Some(choice) = chunk_data.choices.first()
                                && choice.finish_reason.is_some()
                            {
                                final_finish_reason = choice.finish_reason.clone();
                            }
                        }
                    }
                }
            }

            // Flush accumulated tool calls
            for (_index, (id, name, arguments)) in calls {
                if let Ok(arguments_json) = serde_json::from_str::<serde_json::Value>(&arguments) {
                    yield Ok(RawStreamingChoice::ToolCall(
                        RawStreamingToolCall::new(id, name, arguments_json)
                    ));
                }
            }

            // 0.42 replaced the provider-typed final payload with a normalized
            // terminal record: usage is a plain field and the finish reason is
            // reconciled against the tool calls the stream actually produced.
            let mut final_record = StreamFinal::new("blockrun", final_usage.to_rig_usage())
            .with_optional_finish_reason(final_finish_reason.as_deref().map(parse_finish_reason));
            final_record.response_id = final_response_id;
            final_record.model = final_model;

            yield Ok(RawStreamingChoice::FinalResponse(final_record));
        };

        Ok(streaming::StreamingCompletionResponse::stream(
            "blockrun",
            Box::pin(stream),
        ))
    }
}

// ================================================================
// Streaming
// ================================================================

#[derive(Deserialize, Debug)]
struct StreamingDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
    tool_calls: Vec<StreamingToolCall>,
}

#[derive(Deserialize, Debug)]
struct StreamingToolCall {
    #[serde(default)]
    id: Option<String>,
    index: usize,
    function: StreamingFunction,
}

#[derive(Deserialize, Debug)]
struct StreamingFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Deserialize, Debug)]
struct StreamingChoice {
    delta: StreamingDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct StreamingCompletionChunk {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    choices: Vec<StreamingChoice>,
    usage: Option<Usage>,
}

// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_address_derivation() {
        let private_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
        let auth = BlockRunAuth::new(private_key).unwrap();
        assert_eq!(
            auth.address().to_lowercase(),
            "0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266"
        );
    }

    #[test]
    fn test_payment_required_parsing() {
        let json = r#"{
            "x402Version": 2,
            "accepts": [{
                "scheme": "exact",
                "network": "eip155:8453",
                "amount": "1000",
                "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                "payTo": "0x1234567890123456789012345678901234567890",
                "maxTimeoutSeconds": 300,
                "extra": {"name": "USD Coin", "version": "2"}
            }],
            "resource": {
                "url": "https://blockrun.ai/api/v1/chat/completions",
                "description": "AI inference",
                "mimeType": "application/json"
            }
        }"#;

        let payment_required: PaymentRequired = serde_json::from_str(json).unwrap();
        assert_eq!(payment_required.x402_version, 2);
        assert_eq!(payment_required.accepts.len(), 1);
        assert_eq!(payment_required.accepts[0].amount, "1000");
    }

    #[test]
    fn test_completion_response_parsing() {
        let json = r#"{
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello, world!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;

        let response: CompletionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.choices.len(), 1);
        match &response.choices[0].message {
            Message::Assistant { content, .. } => assert_eq!(content, "Hello, world!"),
            _ => panic!("Expected assistant message"),
        }
    }

    /// Anvil's first well-known development key. Public by construction — it is
    /// in every Foundry install — so it is a fixture, not a secret.
    const TEST_KEY: &str = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";

    fn payment_required_fixture() -> PaymentRequired {
        serde_json::from_str(
            r#"{
                "x402Version": 2,
                "accepts": [{
                    "scheme": "exact",
                    "network": "eip155:8453",
                    "amount": "1000",
                    "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                    "payTo": "0x1234567890123456789012345678901234567890",
                    "maxTimeoutSeconds": 300,
                    "extra": {"name": "USD Coin", "version": "2"}
                }]
            }"#,
        )
        .unwrap()
    }

    /// A free-tier client holds no key, so it must not claim an address and
    /// must fail with a directive error rather than panicking when a paid
    /// model reaches the 402.
    #[test]
    fn free_client_has_no_wallet() {
        let client = Client::free().unwrap();
        assert!(client.address().is_none());

        let err = client.signer().unwrap_err().to_string();
        assert!(err.contains("Client::free()"), "unhelpful error: {err}");
    }

    #[test]
    fn wallet_client_exposes_its_address() {
        let client = Client::from_private_key(TEST_KEY).unwrap();
        assert_eq!(
            client.address().map(str::to_lowercase).as_deref(),
            Some("0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266")
        );
    }

    /// Every payment carries the `builder-code` service code, the same
    /// attribution the JS and Python SDKs send. Without it settled payments
    /// cannot be traced back to this SDK.
    #[test]
    fn payment_carries_builder_code() {
        let auth = BlockRunAuth::new(TEST_KEY).unwrap();
        let encoded = auth.create_payment(&payment_required_fixture()).unwrap();
        let decoded = BASE64.decode(&encoded).unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&decoded).unwrap();

        assert_eq!(payload["x402Version"], 2);
        assert_eq!(payload["accepted"]["network"], "eip155:8453");
        assert_eq!(
            payload["extensions"]["builder-code"]["info"]["s"][0],
            "blockrun"
        );
    }

    /// The EIP-712 domain is pinned to USDC on Base and must never be taken
    /// from the server's `extra`, which is attacker-controlled input.
    #[test]
    fn payment_signs_over_the_requested_amount() {
        let auth = BlockRunAuth::new(TEST_KEY).unwrap();
        let encoded = auth.create_payment(&payment_required_fixture()).unwrap();
        let decoded = BASE64.decode(&encoded).unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&decoded).unwrap();

        let authorization = &payload["payload"]["authorization"];
        assert_eq!(authorization["value"], "1000");
        assert_eq!(
            authorization["to"],
            "0x1234567890123456789012345678901234567890"
        );
        assert_eq!(payload["accepted"]["extra"]["name"], "USD Coin");
        assert_eq!(payload["accepted"]["extra"]["version"], "2");
    }

    /// `payTo` arrives from the 402, so a malformed one must surface as an
    /// error. This used to abort the process inside `copy_from_slice`.
    #[test]
    fn malformed_pay_to_is_an_error_not_a_panic() {
        let auth = BlockRunAuth::new(TEST_KEY).unwrap();

        for bad in [
            "0xnothex",
            "0x1234",
            "0x12345678901234567890123456789012345678901234",
            "",
        ] {
            let mut requirements = payment_required_fixture();
            requirements.accepts[0].pay_to = bad.to_string();

            let err = auth
                .create_payment(&requirements)
                .expect_err("expected a rejection for payTo {bad:?}");
            assert!(
                err.to_string().contains("payTo address"),
                "unexpected error for {bad:?}: {err}"
            );
        }
    }

    /// Likewise for the amount: it is server-controlled and was `.parse()`d
    /// with an `expect`.
    #[test]
    fn malformed_amount_is_an_error_not_a_panic() {
        let auth = BlockRunAuth::new(TEST_KEY).unwrap();
        let mut requirements = payment_required_fixture();
        requirements.accepts[0].amount = "not-a-number".to_string();

        let err = auth.create_payment(&requirements).unwrap_err();
        assert!(
            err.to_string().contains("payment amount"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn finish_reasons_normalize_and_carry_through() {
        assert_eq!(parse_finish_reason("stop"), FinishReason::Stop);
        assert_eq!(parse_finish_reason("length"), FinishReason::Length);
        assert_eq!(parse_finish_reason("tool_calls"), FinishReason::ToolCalls);
        assert_eq!(
            parse_finish_reason("content_filter"),
            FinishReason::ContentFilter
        );
        // BlockRun fronts many upstreams; an unknown spelling is carried
        // verbatim rather than silently flattened to `Stop`.
        assert_eq!(
            parse_finish_reason("guardrail_intervened"),
            FinishReason::Other("guardrail_intervened".to_string())
        );
    }

    /// Most of the catalogue is reasoning-capable, so the reasoning and cache
    /// counters have to survive the wire -> rig projection.
    #[test]
    fn usage_projects_cache_and_reasoning_counters() {
        let usage: Usage = serde_json::from_str(
            r#"{
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "total_tokens": 140,
                "prompt_tokens_details": {"cached_tokens": 80},
                "completion_tokens_details": {"reasoning_tokens": 25}
            }"#,
        )
        .unwrap();

        let rig_usage = usage.to_rig_usage();
        assert_eq!(rig_usage.input_tokens, 100);
        assert_eq!(rig_usage.output_tokens, 40);
        assert_eq!(rig_usage.total_tokens, 140);
        assert_eq!(rig_usage.cached_input_tokens, 80);
        assert_eq!(rig_usage.reasoning_tokens, 25);
    }

    /// A gateway that reports no details must leave the counters at zero — the
    /// documented "not reported" sentinel — rather than failing to parse.
    #[test]
    fn usage_without_details_still_parses() {
        let usage: Usage = serde_json::from_str(
            r#"{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}"#,
        )
        .unwrap();

        let rig_usage = usage.to_rig_usage();
        assert_eq!(rig_usage.cached_input_tokens, 0);
        assert_eq!(rig_usage.reasoning_tokens, 0);
    }

    /// rig 0.42 removed `CompletionRequest::preamble`; a system turn now
    /// arrives in `chat_history` and must still reach the wire as a system
    /// message rather than being dropped.
    #[test]
    fn system_history_becomes_a_system_message() {
        let messages = message_to_api(message::Message::System {
            content: "You are terse.".to_string(),
        })
        .unwrap();

        assert_eq!(messages.len(), 1);
        match &messages[0] {
            Message::System { content, .. } => assert_eq!(content, "You are terse."),
            other => panic!("expected a system message, got {other:?}"),
        }
    }

    #[test]
    fn system_instructions_join_every_system_turn() {
        let request = CompletionRequest {
            model: None,
            chat_history: vec![
                message::Message::System {
                    content: "first".to_string(),
                },
                message::Message::user("hello"),
                message::Message::System {
                    content: "second".to_string(),
                },
            ],
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        assert_eq!(
            system_instructions(&request).as_deref(),
            Some("first\nsecond")
        );
    }

    /// The response id, served model, and finish reason all have somewhere to
    /// live in 0.42's flattened response, and the provider's own body survives
    /// in `raw` now that the `CompletionResponse<T>` generic is gone.
    #[test]
    fn response_normalizes_identity_and_keeps_raw() {
        let wire: CompletionResponse = serde_json::from_str(
            r#"{
                "id": "chatcmpl-abc123",
                "model": "anthropic/claude-opus-5",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }"#,
        )
        .unwrap();

        let response: completion::CompletionResponse = wire.try_into().unwrap();

        assert_eq!(response.provider, "blockrun");
        assert_eq!(response.response_id.as_deref(), Some("chatcmpl-abc123"));
        assert_eq!(response.model.as_deref(), Some("anthropic/claude-opus-5"));
        assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
        assert_eq!(response.raw["id"], "chatcmpl-abc123");
    }

    /// A turn that carried tool calls must report `ToolCalls` even when the
    /// gateway labelled it `stop` — OpenAI-compatible fronts routinely do, and
    /// callers branch on this to decide whether to run tools.
    #[test]
    fn tool_call_turn_upgrades_a_plain_stop() {
        let wire: CompletionResponse = serde_json::from_str(
            r#"{
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "add", "arguments": "{\"x\": 1, \"y\": 2}"}
                        }]
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }"#,
        )
        .unwrap();

        let response: completion::CompletionResponse = wire.try_into().unwrap();
        assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
    }

    /// An assistant turn with neither text nor a tool call is a provider
    /// defect; it must be rejected rather than yielding an empty choice.
    #[test]
    fn empty_assistant_turn_is_rejected() {
        let wire: CompletionResponse = serde_json::from_str(
            r#"{
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}
            }"#,
        )
        .unwrap();

        let err = completion::CompletionResponse::try_from(wire).unwrap_err();
        assert!(err.to_string().contains("empty"), "unexpected error: {err}");
    }
}
