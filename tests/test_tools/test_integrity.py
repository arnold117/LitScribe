from litscribe.tools.integrity import sign_finding, verify_finding, sign_record, verify_record


def test_sign_and_verify():
    sig = sign_finding("Biology", "CRISPR", "FUT8 knockout")
    assert verify_finding("Biology", "CRISPR", "FUT8 knockout", sig)


def test_tamper_detection():
    sig = sign_finding("Biology", "CRISPR", "FUT8 knockout")
    assert not verify_finding("Biology", "CRISPR", "HACKED content", sig)


def test_different_domains():
    sig1 = sign_finding("Biology", "t", "f")
    sig2 = sign_finding("Chemistry", "t", "f")
    assert sig1 != sig2


def test_record_sign():
    sig = sign_record("review text here")
    assert verify_record("review text here", sig)
    assert not verify_record("tampered text", sig)
