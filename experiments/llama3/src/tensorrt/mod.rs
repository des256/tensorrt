use std::time::Instant;
use tensorrt::{Executor, SamplingParams};
use tokenizers::Tokenizer;

const BOS_TOKEN_ID: i32 = 128000; // <|begin_of_text|>
const EOS_TOKEN_ID: i32 = 128001; // <|end_of_text|>
const EOT_TOKEN_ID: i32 = 128009; // <|eot_id|>
const MAX_NEW_TOKENS: i32 = 512;

const HEADER_START: &str = "<|start_header_id|>";
const HEADER_END: &str = "<|end_header_id|>";
const EOT: &str = "<|eot_id|>";

fn apply_chat_template(prompt: &str) -> String {
    format!(
        "{HEADER_START}system{HEADER_END}\n\nYou are a helpful assistant.{EOT}\
         {HEADER_START}user{HEADER_END}\n\n{prompt}{EOT}\
         {HEADER_START}assistant{HEADER_END}\n\n"
    )
}

pub fn generate(
    executor: &Executor,
    tokenizer: &Tokenizer,
    prompt: &str,
) -> (String, u128) {
    let templated = apply_chat_template(prompt);

    let encoding = tokenizer
        .encode(templated.as_str(), false)
        .unwrap_or_else(|e| panic!("Failed to encode prompt: {e}"));

    // Prepend BOS token
    let mut input_ids: Vec<i32> = Vec::with_capacity(1 + encoding.get_ids().len());
    input_ids.push(BOS_TOKEN_ID);
    input_ids.extend(encoding.get_ids().iter().map(|&id| id as i32));

    let sampling = SamplingParams::default();

    let start = Instant::now();

    let req_id = executor.enqueue(
        &input_ids,
        MAX_NEW_TOKENS,
        &sampling,
        Some(EOT_TOKEN_ID),
        Some(EOS_TOKEN_ID),
        true,
    );

    let mut all_tokens: Vec<i32> = Vec::new();
    let mut time_to_first_token_ms = 0u128;

    loop {
        let (tokens, is_final) = executor.await_response(req_id, 0);
        if !tokens.is_empty() && time_to_first_token_ms == 0 {
            time_to_first_token_ms = start.elapsed().as_millis();
        }
        all_tokens.extend_from_slice(&tokens);
        if is_final {
            break;
        }
    }

    // Strip stop tokens from output if present
    while let Some(&last) = all_tokens.last() {
        if last == EOS_TOKEN_ID || last == EOT_TOKEN_ID {
            all_tokens.pop();
        } else {
            break;
        }
    }

    let output_ids: Vec<u32> = all_tokens.iter().map(|&id| id as u32).collect();
    let text = tokenizer
        .decode(&output_ids, true)
        .unwrap_or_else(|e| panic!("Failed to decode tokens: {e}"));

    (text, time_to_first_token_ms)
}
