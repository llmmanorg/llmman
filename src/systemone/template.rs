//! The model's own Jinja chat template, rendered the way llama-server does
//! for one user turn: HF's `trim_blocks`/`lstrip_blocks`, Python's string
//! methods, `raise_exception` and `strftime_now`.

use anyhow::{anyhow, Result};
use minijinja::{Environment, Error, ErrorKind, Value};
use serde_json::{json, Map, Value as Json};

pub struct Template {
    env: Environment<'static>,
    bos: String,
    eos: String,
}

impl Template {
    /// `bos` and `eos` are the vocabulary's token texts, `""` when it has none.
    pub fn new(source: &str, bos: &str, eos: &str) -> Result<Template> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_function("strftime_now", |fmt: String| {
            chrono::Local::now().format(&fmt).to_string()
        });
        env.add_template_owned("chat", source.to_string())
            .map_err(|e| anyhow!("the chat template does not parse: {e:#}"))?;
        Ok(Template {
            env,
            bos: bos.into(),
            eos: eos.into(),
        })
    }

    /// One user message and the assistant turn opener. `kwargs` are the
    /// template variables (`enable_thinking`, ...).
    pub fn render(&self, content: &str, kwargs: &Map<String, Json>) -> Result<String> {
        let mut vars = kwargs.clone();
        vars.insert(
            "messages".into(),
            json!([{"role": "user", "content": content}]),
        );
        vars.insert("add_generation_prompt".into(), true.into());
        vars.insert("bos_token".into(), self.bos.clone().into());
        vars.insert("eos_token".into(), self.eos.clone().into());
        self.env
            .get_template("chat")
            .and_then(|t| t.render(Value::from_serialize(&vars)))
            .map_err(|e| anyhow!("the chat template failed: {e:#}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape of Qwen3.5's template around the generation prompt.
    const QWEN_LIKE: &str = "{%- for message in messages %}\n\
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}\n\
        {%- endfor %}\n\
        {%- if add_generation_prompt %}\n\
            {{- '<|im_start|>assistant\\n' }}\n\
            {%- if enable_thinking is defined and enable_thinking is true %}\n\
                {{- '<think>\\n' }}\n\
            {%- else %}\n\
                {{- '<think>\\n\\n</think>\\n\\n' }}\n\
            {%- endif %}\n\
        {%- endif %}";

    fn kwargs(json: &str) -> Map<String, Json> {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn the_thinking_variable_picks_the_generation_prompt() {
        let t = Template::new(QWEN_LIKE, "", "<|im_end|>").unwrap();
        let off = t
            .render("hi", &kwargs(r#"{"enable_thinking": false}"#))
            .unwrap();
        assert_eq!(
            off,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let on = t
            .render("hi", &kwargs(r#"{"enable_thinking": true}"#))
            .unwrap();
        assert!(on.ends_with("<|im_start|>assistant\n<think>\n"), "{on:?}");
    }

    #[test]
    fn templates_get_the_special_tokens_and_pythons_string_methods() {
        let t = Template::new(
            "{{ bos_token }}{{ messages[0]['content'].strip().startswith('a') }}{{ eos_token }}",
            "<bos>",
            "<eos>",
        )
        .unwrap();
        assert_eq!(t.render("  abc ", &Map::new()).unwrap(), "<bos>True<eos>");
    }

    #[test]
    fn blocks_trim_like_hugging_face() {
        let t = Template::new("{% if true %}\n  x\n{% endif %}\ny", "", "").unwrap();
        assert_eq!(t.render("", &Map::new()).unwrap(), "  x\ny");
    }

    #[test]
    fn a_template_that_raises_or_does_not_parse_is_an_error() {
        let t = Template::new("{{ raise_exception('no system role') }}", "", "").unwrap();
        let e = t.render("", &Map::new()).unwrap_err().to_string();
        assert!(e.contains("no system role"), "{e}");
        assert!(Template::new("{% if %}", "", "").is_err());
    }
}
