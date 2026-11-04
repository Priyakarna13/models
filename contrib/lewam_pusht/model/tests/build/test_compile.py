"""Stage-output acceptance: candidate outputs and raw ELFs are never sufficient."""

import json

from model_registry.artifact import validate_compile_outputs


def test_all_four_standard_components(compiled, build_profile):
    report = validate_compile_outputs(
        compiled / "compile-outputs.json", compiled, model_id="lewam_pusht",
        build_profile_id=build_profile, compile_invocation_id=build_profile,
        component_ids=["front", "body", "head", "policy"],
    )
    assert len(report["outputs"]) == 4
    # A future qualification producer must bind exact final MPK/ELF identities;
    # false compile options or old campaign certificates are not evidence.
    qualification = json.loads((compiled / "qualification.json").read_text())
    assert qualification["status"] == "PASS"
    assert qualification["model_id"] == "lewam_pusht"
    assert qualification["build_profile_id"] == build_profile
    assert set(qualification["components"]) == {"front", "body", "head", "policy"}
    for row in report["outputs"]:
        proof = qualification["components"][row["component_id"]]
        assert proof["mpk_sha256"] == row["mpk_sha256"]
        assert proof["no_fp32_mla"] is True
        assert proof["public_neat_native_parity"] is True
        assert proof["source_equivalence"] is True
