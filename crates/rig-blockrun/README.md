## Rig-BlockRun

This companion crate integrates [BlockRun](https://blockrun.ai) with Rig, providing pay-per-request access to ~100 chat models through the BlockRun account API or x402 micropayments.

### Features

- **Account API**: create a key at [user.blockrun.ai](https://user.blockrun.ai/dashboard/keys) and bill requests to account credits
- **x402 Protocol**: alternatively use signed USDC authorizations on Solana or Base
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

For paid models, create an account at [user.blockrun.ai](https://user.blockrun.ai),
add [credits](https://user.blockrun.ai/dashboard/credits), create an
[API key](https://user.blockrun.ai/dashboard/keys), and set:

```bash
export BLOCKRUN_API_KEY=brk_...
```

`Client::from_env()` uses the account key first. The existing x402 wallet flow remains
available through `BLOCKRUN_WALLET_KEY`; BlockRun recommends Solana before Base for
wallet payments. Credentials stay local and are never printed by the client.

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

    // Paid models: account API key first, wallet fallback.
    let client = Client::from_env()?;
    println!("Billing: {}", client.auth_mode());

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
