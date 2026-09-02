//! Smoke test against the live free tier — no wallet, no payment.
//!
//! Exercises the three paths that matter: unary completion, streaming, and
//! tool calling.
//!
//! ```bash
//! cargo run -p rig-blockrun --example free_tier
//! ```

use futures::StreamExt;
use rig_agent::prelude::*;
use rig_agent::tool::ToolContext;
use rig_blockrun::{Client, FREE_QWEN35_397B};
use rig_core::completion::CompletionModel;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = Client::free()?;
    println!("wallet: {:?}\n", client.address());

    // ---- 1. unary ----
    println!("--- unary ---");
    let agent = client
        .agent(FREE_QWEN35_397B)
        .preamble("Answer in one short sentence. Do not think out loud.")
        .build();
    let response = agent.prompt("What is 2 + 2?").await?;
    println!("{}\n", response.output);

    // ---- 2. streaming ----
    println!("--- streaming ---");
    let model = client.completion_model(FREE_QWEN35_397B);
    let request = model
        .completion_request("Count from 1 to 5, separated by spaces.")
        .build();
    let mut stream = model.stream(request).await?;

    let mut chunks = 0usize;
    while let Some(item) = stream.next().await {
        if let Ok(content) = item {
            chunks += 1;
            print!("{content:?} ");
        }
    }
    println!("\n[{chunks} stream items]");
    println!(
        "final usage: {:?}\n",
        stream.response.as_ref().map(|r| &r.usage)
    );

    // ---- 3. tool calling ----
    println!("--- tool calling ---");
    let calculator = client
        .agent(FREE_QWEN35_397B)
        .preamble("You are a calculator. Always use the provided tool.")
        .max_tokens(512)
        .tool(Adder)
        .build();
    let response = calculator.prompt("What is 15 + 27?").max_turns(3).await?;
    println!("{}", response.output);

    Ok(())
}

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
        println!("[tool called] add({}, {})", args.x, args.y);
        Ok(args.x + args.y)
    }
}
