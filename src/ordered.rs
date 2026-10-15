//! A JSON object that keeps its members in order, which serde_json's own
//! map does not. `/v1/systemone` needs it end to end: questions are read in
//! the order written and answers come back in it.

use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

/// An object's members, in order; with `Box<RawValue>` they pass through
/// byte for byte.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Ordered<T>(pub Vec<(String, T)>);

impl<T> Ordered<T> {
    pub fn get(&self, key: &str) -> Option<&T> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn remove(&mut self, key: &str) -> Option<T> {
        let at = self.0.iter().position(|(k, _)| k == key)?;
        Some(self.0.remove(at).1)
    }

    /// Replaces `key` where it stands, or adds it last.
    pub fn set(&mut self, key: &str, value: T) {
        match self.0.iter_mut().find(|(k, _)| k == key) {
            Some((_, v)) => *v = value,
            None => self.0.push((key.to_string(), value)),
        }
    }
}

impl<T: Serialize> Serialize for Ordered<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.0.iter().map(|(key, value)| (key, value)))
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Ordered<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Members<T>(PhantomData<T>);

        impl<'de, T: Deserialize<'de>> serde::de::Visitor<'de> for Members<T> {
            type Value = Ordered<T>;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("an object")
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let mut members = Vec::new();
                while let Some(member) = map.next_entry()? {
                    members.push(member);
                }
                Ok(Ordered(members))
            }
        }

        deserializer.deserialize_map(Members(PhantomData))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::value::RawValue;

    #[test]
    fn members_keep_their_order_and_their_bytes() {
        let text = r#"{"z":1,"a":{"y":[1, 2],"b":"x"},"m":null}"#;
        let object: Ordered<Box<RawValue>> = serde_json::from_str(text).unwrap();
        let keys: Vec<_> = object.0.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["z", "a", "m"]);
        // The raw member is exactly as written: spacing and inner order.
        assert_eq!(object.get("a").unwrap().get(), r#"{"y":[1, 2],"b":"x"}"#);
        assert_eq!(serde_json::to_string(&object).unwrap(), text);
    }

    #[test]
    fn set_replaces_in_place_or_appends_and_remove_takes_out() {
        let mut object: Ordered<i32> = Ordered(vec![("a".into(), 1), ("b".into(), 2)]);
        object.set("a", 10);
        object.set("c", 3);
        assert_eq!(
            serde_json::to_string(&object).unwrap(),
            r#"{"a":10,"b":2,"c":3}"#
        );
        assert_eq!(object.remove("b"), Some(2));
        assert_eq!(object.remove("b"), None);
        assert_eq!(serde_json::to_string(&object).unwrap(), r#"{"a":10,"c":3}"#);
    }

    #[test]
    fn only_an_object_is_one() {
        assert!(serde_json::from_str::<Ordered<i32>>("[1]").is_err());
    }
}
