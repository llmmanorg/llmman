//! Signed JWTs (RS256 and ES256, via `ring`) for a Google service account's
//! token request and `jwt_bearer` providers ([`crate::tokensvc`]). Keys are
//! PKCS#8 or PKCS#1; a SEC1 EC key is refused with the conversion command.

use anyhow::{anyhow, bail, Context};
use base64::Engine as _;
use ring::rand::SystemRandom;
use ring::signature::{EcdsaKeyPair, RsaKeyPair, ECDSA_P256_SHA256_FIXED_SIGNING};
use serde_json::{json, Value};

/// A JWS algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alg {
    Rs256,
    Es256,
}

impl Alg {
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim() {
            "RS256" => Some(Self::Rs256),
            "ES256" => Some(Self::Es256),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rs256 => "RS256",
            Self::Es256 => "ES256",
        }
    }
}

enum Pair {
    Rsa(RsaKeyPair),
    Ec(EcdsaKeyPair),
}

/// A private key that signs JWTs. No `Debug` of the key material.
pub struct SigningKey {
    pair: Pair,
}

impl std::fmt::Debug for SigningKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SigningKey({})", self.alg().as_str())
    }
}

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

/// The DER inside one PEM block, and its label.
fn pem_block(pem: &str) -> anyhow::Result<(String, Vec<u8>)> {
    let begin = pem
        .find("-----BEGIN ")
        .ok_or_else(|| anyhow!("not a PEM private key"))?;
    let rest = &pem[begin + "-----BEGIN ".len()..];
    let label_end = rest
        .find("-----")
        .ok_or_else(|| anyhow!("not a PEM private key"))?;
    let label = rest[..label_end].to_string();
    let body_start = label_end + 5;
    let end_marker = format!("-----END {label}-----");
    let body_end = rest[body_start..]
        .find(&end_marker)
        .ok_or_else(|| anyhow!("the PEM private key is not closed"))?;
    let body: String = rest[body_start..body_start + body_end]
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    let der = base64::engine::general_purpose::STANDARD
        .decode(body)
        .map_err(|_| anyhow!("the PEM private key is not valid base64"))?;
    Ok((label, der))
}

impl SigningKey {
    /// Reads a PEM private key. `hint` says which algorithm it is for when
    /// the file does not (a PKCS#8 block holds either kind of key).
    pub fn from_pem(pem: &str, hint: Option<Alg>) -> anyhow::Result<Self> {
        let (label, der) = pem_block(pem)?;
        match label.as_str() {
            "RSA PRIVATE KEY" => RsaKeyPair::from_der(&der)
                .map(|k| Self { pair: Pair::Rsa(k) })
                .map_err(|e| anyhow!("not a usable RSA private key: {e}")),
            "EC PRIVATE KEY" => bail!(
                "a SEC1 EC key is not supported; convert it with \
                 `openssl pkcs8 -topk8 -nocrypt -in key.pem -out key-pkcs8.pem`"
            ),
            "PRIVATE KEY" => {
                let rsa = || RsaKeyPair::from_pkcs8(&der).map(Pair::Rsa);
                let ec = || {
                    EcdsaKeyPair::from_pkcs8(
                        &ECDSA_P256_SHA256_FIXED_SIGNING,
                        &der,
                        &SystemRandom::new(),
                    )
                    .map(Pair::Ec)
                };
                let pair = match hint {
                    Some(Alg::Es256) => ec().ok().or_else(|| rsa().ok()),
                    _ => rsa().ok().or_else(|| ec().ok()),
                };
                pair.map(|pair| Self { pair }).ok_or_else(|| {
                    anyhow!("not a usable RSA or P-256 private key (PKCS#8, unencrypted)")
                })
            }
            other => bail!("a {other:?} block is not a private key"),
        }
    }

    pub fn alg(&self) -> Alg {
        match self.pair {
            Pair::Rsa(_) => Alg::Rs256,
            Pair::Ec(_) => Alg::Es256,
        }
    }

    fn sign(&self, message: &[u8]) -> anyhow::Result<Vec<u8>> {
        match &self.pair {
            Pair::Rsa(key) => {
                let mut signature = vec![0u8; key.public().modulus_len()];
                key.sign(
                    &ring::signature::RSA_PKCS1_SHA256,
                    &SystemRandom::new(),
                    message,
                    &mut signature,
                )
                .map_err(|_| anyhow!("RSA signing failed"))?;
                Ok(signature)
            }
            Pair::Ec(key) => key
                .sign(&SystemRandom::new(), message)
                .map(|s| s.as_ref().to_vec())
                .map_err(|_| anyhow!("ECDSA signing failed")),
        }
    }

    /// `claims` as a compact JWS. `kid` goes in the header when given.
    pub fn sign_jwt(&self, kid: Option<&str>, claims: &Value) -> anyhow::Result<String> {
        let mut header = json!({ "alg": self.alg().as_str(), "typ": "JWT" });
        if let Some(kid) = kid {
            header["kid"] = json!(kid);
        }
        let signing_input = format!(
            "{}.{}",
            b64(&serde_json::to_vec(&header).context("serialize the JWT header")?),
            b64(&serde_json::to_vec(claims).context("serialize the JWT claims")?)
        );
        let signature = self.sign(signing_input.as_bytes())?;
        Ok(format!("{signing_input}.{}", b64(&signature)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ring::signature::{UnparsedPublicKey, ECDSA_P256_SHA256_FIXED, RSA_PKCS1_2048_8192_SHA256};

    const RSA_PKCS8: &str = include_str!("testdata/jwt-test-rsa.pem");
    const RSA_PKCS1: &str = include_str!("testdata/jwt-test-rsa-pkcs1.pem");

    fn split(jwt: &str) -> (String, Vec<u8>, Value, Value) {
        let parts: Vec<&str> = jwt.split('.').collect();
        assert_eq!(parts.len(), 3, "{jwt}");
        let dec = |p: &str| {
            base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(p)
                .unwrap()
        };
        (
            format!("{}.{}", parts[0], parts[1]),
            dec(parts[2]),
            serde_json::from_slice(&dec(parts[0])).unwrap(),
            serde_json::from_slice(&dec(parts[1])).unwrap(),
        )
    }

    #[test]
    fn rs256_signs_with_a_pkcs8_or_a_pkcs1_key_and_verifies() {
        for pem in [RSA_PKCS8, RSA_PKCS1] {
            let key = SigningKey::from_pem(pem, None).unwrap();
            assert_eq!(key.alg(), Alg::Rs256);
            let jwt = key
                .sign_jwt(
                    Some("key-1"),
                    &json!({ "iss": "me", "aud": "you", "exp": 5 }),
                )
                .unwrap();
            let (input, sig, header, claims) = split(&jwt);
            assert_eq!(
                header,
                json!({ "alg": "RS256", "typ": "JWT", "kid": "key-1" })
            );
            assert_eq!(claims["iss"], "me");
            let public = match &key.pair {
                Pair::Rsa(k) => k.public().as_ref().to_vec(),
                Pair::Ec(_) => unreachable!(),
            };
            UnparsedPublicKey::new(&RSA_PKCS1_2048_8192_SHA256, public)
                .verify(input.as_bytes(), &sig)
                .expect("a valid RS256 signature");
        }
    }

    #[test]
    fn es256_signs_a_p256_pkcs8_key_and_verifies() {
        let generated = rcgen::KeyPair::generate().unwrap();
        let key = SigningKey::from_pem(&generated.serialize_pem(), Some(Alg::Es256)).unwrap();
        assert_eq!(key.alg(), Alg::Es256);
        let jwt = key.sign_jwt(None, &json!({ "sub": "s" })).unwrap();
        let (input, sig, header, _) = split(&jwt);
        assert_eq!(header, json!({ "alg": "ES256", "typ": "JWT" }));
        assert_eq!(sig.len(), 64, "a fixed-width r||s signature");
        UnparsedPublicKey::new(&ECDSA_P256_SHA256_FIXED, generated.public_key_raw())
            .verify(input.as_bytes(), &sig)
            .expect("a valid ES256 signature");
    }

    #[test]
    fn unusable_keys_are_refused_without_echoing_them() {
        for (pem, wants) in [
            ("not a key", "not a PEM"),
            (
                "-----BEGIN EC PRIVATE KEY-----\nAAAA\n-----END EC PRIVATE KEY-----",
                "SEC1",
            ),
            (
                "-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----",
                "not a private key",
            ),
            (
                "-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----",
                "not a usable",
            ),
            ("-----BEGIN PRIVATE KEY-----\nAA", "not closed"),
        ] {
            let err = SigningKey::from_pem(pem, None).unwrap_err().to_string();
            assert!(err.contains(wants), "{pem}: {err}");
        }
        assert_eq!(
            format!("{:?}", SigningKey::from_pem(RSA_PKCS8, None).unwrap()),
            "SigningKey(RS256)"
        );
        assert_eq!(Alg::parse("ES256"), Some(Alg::Es256));
        assert_eq!(Alg::parse("HS256"), None);
    }
}
