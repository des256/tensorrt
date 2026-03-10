use std::time::Instant;

pub use onnx::*;
use tokenizers::Tokenizer;

const NUM_LAYERS: usize = 28;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const VOCAB_SIZE: usize = 128256;

const BOS_TOKEN_ID: i64 = 128000;
const EOS_TOKEN_ID: i64 = 128001;
const EOT_TOKEN_ID: i64 = 128009;
const MAX_NEW_TOKENS: usize = 512;

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

fn argmax_logits(logits_value: &Value, seq_len: usize) -> i64 {
    let logits_f32 = logits_value.extract_as_f32();
    let offset = (seq_len - 1) * VOCAB_SIZE;
    let last_logits = &logits_f32[offset..offset + VOCAB_SIZE];

    let mut max_idx = 0usize;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in last_logits.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx as i64
}

pub fn generate(session: &Session, tokenizer: &Tokenizer, prompt: &str) -> (String, u128) {
    let onnx = &session.onnx;
    let templated = apply_chat_template(prompt);

    let encoding = tokenizer
        .encode(templated.as_str(), false)
        .unwrap_or_else(|e| panic!("Failed to encode prompt: {e}"));

    let mut input_token_ids: Vec<i64> = Vec::with_capacity(1 + encoding.get_ids().len());
    input_token_ids.push(BOS_TOKEN_ID);
    input_token_ids.extend(encoding.get_ids().iter().map(|&id| id as i64));

    let prompt_len = input_token_ids.len();

    // Build KV cache input/output name strings
    let kv_input_names: Vec<String> = (0..NUM_LAYERS)
        .flat_map(|i| {
            [
                format!("past_key_values.{i}.key"),
                format!("past_key_values.{i}.value"),
            ]
        })
        .collect();

    let mut all_output_names: Vec<String> = Vec::with_capacity(1 + NUM_LAYERS * 2);
    all_output_names.push("logits".to_string());
    for i in 0..NUM_LAYERS {
        all_output_names.push(format!("present.{i}.key"));
        all_output_names.push(format!("present.{i}.value"));
    }
    let output_name_refs: Vec<&str> = all_output_names.iter().map(|s| s.as_str()).collect();

    // Detect the KV cache element type from the model's first past_key_values input
    let kv_dtype = {
        let input_count = session.input_count();
        let mut dtype = ONNXTensorElementDataType::Float;
        for idx in 0..input_count {
            if session.input_name(idx).starts_with("past_key_values") {
                dtype = session.input_element_type(idx);
                break;
            }
        }
        dtype
    };

    // Initial empty KV cache: [1, 8, 0, 128]
    let mut kv_cache: Vec<Value> = (0..NUM_LAYERS * 2)
        .map(|_| Value::empty_typed(onnx, &[1, NUM_KV_HEADS, 0, HEAD_DIM], kv_dtype))
        .collect();

    let start = Instant::now();
    let mut all_generated_tokens: Vec<i64> = Vec::new();

    // --- Prefill pass ---
    let input_ids_val = Value::from_slice(onnx, &[1, prompt_len], &input_token_ids);
    let attention_mask: Vec<i64> = vec![1i64; prompt_len];
    let attention_mask_val = Value::from_slice(onnx, &[1, prompt_len], &attention_mask);
    let position_ids: Vec<i64> = (0..prompt_len as i64).collect();
    let position_ids_val = Value::from_slice(onnx, &[1, prompt_len], &position_ids);

    let mut inputs: Vec<(&str, &Value)> = Vec::with_capacity(3 + NUM_LAYERS * 2);
    inputs.push(("input_ids", &input_ids_val));
    inputs.push(("attention_mask", &attention_mask_val));
    inputs.push(("position_ids", &position_ids_val));
    for (i, kv) in kv_cache.iter().enumerate() {
        inputs.push((&kv_input_names[i], kv));
    }

    let outputs = session.run(&inputs, &output_name_refs);

    let next_token = argmax_logits(&outputs[0], prompt_len);
    let time_to_first_token_ms = start.elapsed().as_millis();

    let mut past_seq_len = prompt_len;
    kv_cache = outputs.into_iter().skip(1).collect();

    if next_token != EOS_TOKEN_ID && next_token != EOT_TOKEN_ID {
        all_generated_tokens.push(next_token);
    }

    // --- Decode loop ---
    while all_generated_tokens.len() < MAX_NEW_TOKENS {
        let last_token = match all_generated_tokens.last() {
            Some(&t) => t,
            None => break,
        };

        let input_ids_val = Value::from_slice(onnx, &[1, 1], &[last_token]);
        let attention_mask: Vec<i64> = vec![1i64; past_seq_len + 1];
        let attention_mask_val = Value::from_slice(onnx, &[1, past_seq_len + 1], &attention_mask);
        let position_ids_val = Value::from_slice(onnx, &[1, 1], &[past_seq_len as i64]);

        let mut inputs: Vec<(&str, &Value)> = Vec::with_capacity(3 + NUM_LAYERS * 2);
        inputs.push(("input_ids", &input_ids_val));
        inputs.push(("attention_mask", &attention_mask_val));
        inputs.push(("position_ids", &position_ids_val));
        for (i, kv) in kv_cache.iter().enumerate() {
            inputs.push((&kv_input_names[i], kv));
        }

        let outputs = session.run(&inputs, &output_name_refs);

        let next_token = argmax_logits(&outputs[0], 1);

        past_seq_len += 1;
        kv_cache = outputs.into_iter().skip(1).collect();

        if next_token == EOS_TOKEN_ID || next_token == EOT_TOKEN_ID {
            break;
        }
        all_generated_tokens.push(next_token);
    }

    // Strip trailing stop tokens
    while let Some(&last) = all_generated_tokens.last() {
        if last == EOS_TOKEN_ID || last == EOT_TOKEN_ID {
            all_generated_tokens.pop();
        } else {
            break;
        }
    }

    let output_ids: Vec<u32> = all_generated_tokens.iter().map(|&id| id as u32).collect();
    let text = tokenizer
        .decode(&output_ids, true)
        .unwrap_or_else(|e| panic!("Failed to decode tokens: {e}"));

    (text, time_to_first_token_ms)
}
