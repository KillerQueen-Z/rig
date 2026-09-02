## Rig-BlockRun

This companion crate integrates [BlockRun](https://blockrun.ai) with Rig, providing pay-per-request access to ~100 chat models via x402 micropayments.

### Features

- **No API keys**: wallet signatures replace API keys; nothing to provision, no account to create
- **x402 Protocol**: HTTP 402 Payment Required with EIP-712 signed USDC authorizations on Base
- **Free tier**: part of the catalogue costs nothing and needs no wallet at all
- **Multi-model access**: Claude, GPT-5.6, Gemini, Grok, DeepSeek, Kimi, GLM, Qwen, MiniMax and more behind one provider
- **Tool calling and streaming**: full compatibility with Rig's tool and agent system

## Usage

Add the companion crate to your `Cargo.toml`, along with the rig-core crate:

```toml
[dependencies]
rig-blockrun = "0.42.0"
rig-core = "0.42.0"
```

You can also run `cargo add rig-blockrun rig-core` to add the most recent versions of the dependencies to your project.

### Setup

Free models need no setup at all. For the paid catalogue:

1. Generate a wallet private key or use an existing one
2. Fund it with USDC on Base (even $1 goes a long way — requests start around $0.002)
3. Set the `BLOCKRUN_WALLET_KEY` environment variable

The private key is only ever used for local signing; it never leaves your machine.

### Example

```rust
use rig_agent::prelude::*;
use rig_blockrun::{Client, CLAUDE_OPUS_5, FREE_QWEN35_397B};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Free tier: no wallet, no payment.
    let free_agent = Client::free()?
        .agent(FREE_QWEN35_397B)
        .preamble("You are a helpful assistant.")
        .build();

    println!("{}", free_agent.prompt("What is x402?").await?.output);

    // Paid models: one wallet, every provider.
    let client = Client::from_env()?;

    if let Some(address) = client.address() {
        println!("Wallet: {address}");
    }

    let agent = client
        .agent(CLAUDE_OPUS_5)
        .preamble("You are a helpful assistant.")
        .build();

    println!("{}", agent.prompt("What is x402?").await?.output);

    Ok(())
}
```

### Model ids

The constants in this crate are a convenience, not a limit — any id the gateway
serves works as a plain string, and `GET /v1/models` is the live catalogue:

```rust
let model = client.completion_model("zai/glm-5.3");
```

Constants are provided for the current flagships across Anthropic, OpenAI,
Google, xAI, DeepSeek, Moonshot, Z.ai, MiniMax and Qwen, plus the `FREE_*`
models that skip payment entirely.

See the [`/examples`](./examples) folder for more usage examples.
