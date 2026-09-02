//! Example of using the BlockRun provider with Rig.
//!
//! BlockRun provides pay-per-request access to ~100 chat models via x402
//! micropayments. No API keys — just a wallet funded with USDC on Base. Part of
//! the catalogue is free and needs no wallet at all.
//!
//! # Setup
//!
//! 1. Generate a wallet private key or use an existing one
//! 2. Fund it with USDC on Base (even $1 goes a long way at ~$0.002/request)
//! 3. Set `BLOCKRUN_WALLET_KEY`
//!
//! The free-tier section below runs without any of that.
//!
//! # Running
//!
//! ```bash
//! BLOCKRUN_WALLET_KEY=0x... cargo run -p rig-blockrun --example agent_with_blockrun
//! ```

use rig_agent::prelude::*;
use rig_agent::tool::ToolContext;
use rig_blockrun::{CLAUDE_OPUS_5, Client, DEEPSEEK_V4_PRO, FREE_QWEN35_397B, GPT_56_TERRA};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    // ---- Free tier: no wallet, no payment ----
    println!("=== Free tier (no wallet) ===");
    let free_agent = Client::free()?
        .agent(FREE_QWEN35_397B)
        .preamble("You are a helpful assistant.")
        .build();

    let answer = free_agent
        .prompt("What is x402 in one sentence?")
        .await?
        .output;
    println!("Free model: {answer}\n");

    // ---- Paid models: sign with a funded wallet ----
    let client = Client::from_env()?;

    // Handy when you need to top the wallet up.
    if let Some(address) = client.address() {
        println!("Wallet address: {address}");
        println!("Fund it with USDC on Base to use the paid models\n");
    }

    println!("=== Claude Opus 5 ===");
    let claude_agent = client
        .agent(CLAUDE_OPUS_5)
        .preamble("You are a helpful assistant.")
        .build();

    let answer = claude_agent
        .prompt("What is x402 in one sentence?")
        .await?
        .output;
    println!("Claude: {answer}\n");

    println!("=== GPT-5.6 Terra ===");
    let gpt_agent = client
        .agent(GPT_56_TERRA)
        .preamble("You are a helpful assistant.")
        .build();

    let answer = gpt_agent
        .prompt("What is x402 in one sentence?")
        .await?
        .output;
    println!("GPT-5.6: {answer}\n");

    println!("=== DeepSeek V4 Pro (cost-effective) ===");
    let deepseek_agent = client
        .agent(DEEPSEEK_V4_PRO)
        .preamble("You are a helpful assistant.")
        .build();

    let answer = deepseek_agent
        .prompt("What is x402 in one sentence?")
        .await?
        .output;
    println!("DeepSeek: {answer}\n");

    // ---- Tool calling ----
    println!("=== Calculator agent with tools ===");
    let calculator_agent = client
        .agent(CLAUDE_OPUS_5)
        .preamble("You are a calculator. Use the provided tools to perform calculations.")
        .max_tokens(1024)
        .tool(Adder)
        .tool(Multiplier)
        .build();

    let answer = calculator_agent
        .prompt("What is (15 + 7) * 3?")
        .await?
        .output;
    println!("Calculator: {answer}");

    Ok(())
}

// Tool definitions

#[derive(Deserialize, JsonSchema)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
struct MathError;

#[derive(Deserialize, Serialize)]
struct Adder;

impl Tool for Adder {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Add two numbers together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!(schema_for!(OperationArgs))
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        println!("[tool] add({}, {}) = {}", args.x, args.y, args.x + args.y);
        Ok(args.x + args.y)
    }
}

#[derive(Deserialize, Serialize)]
struct Multiplier;

impl Tool for Multiplier {
    const NAME: &'static str = "multiply";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Multiply two numbers together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!(schema_for!(OperationArgs))
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        println!(
            "[tool] multiply({}, {}) = {}",
            args.x,
            args.y,
            args.x * args.y
        );
        Ok(args.x * args.y)
    }
}
