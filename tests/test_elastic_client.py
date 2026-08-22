import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from elastic_client import ssl_client_kwargs


class ElasticClientSslTests(unittest.TestCase):
    def test_uses_explicit_ca_path(self):
        with tempfile.TemporaryDirectory() as raw:
            cert = Path(raw) / "ca.crt"
            cert.write_text("dummy-cert\n", encoding="utf-8")
            with patch.dict(os.environ, {"ELASTIC_CA_CERT": str(cert), "ELASTIC_VERIFY_CERTS": ""}, clear=False):
                kwargs = ssl_client_kwargs("ca.crt")
        self.assertTrue(kwargs["verify_certs"])
        self.assertEqual(kwargs["ca_certs"], str(cert))

    def test_missing_cert_skips_verification(self):
        with patch.dict(os.environ, {"ELASTIC_CA_CERT": "", "ELASTIC_VERIFY_CERTS": ""}, clear=False):
            with patch("elastic_client.resolve_ca_cert", return_value=None):
                kwargs = ssl_client_kwargs("ca.crt")
        self.assertFalse(kwargs["verify_certs"])
        self.assertNotIn("ca_certs", kwargs)

    def test_env_can_disable_verification(self):
        with patch.dict(os.environ, {"ELASTIC_VERIFY_CERTS": "false"}, clear=False):
            kwargs = ssl_client_kwargs("ca.crt")
        self.assertFalse(kwargs["verify_certs"])
        self.assertNotIn("ca_certs", kwargs)


if __name__ == "__main__":
    unittest.main()
