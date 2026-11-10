"""Independent regressions at the public operation and certificate boundary."""

from copy import deepcopy

import pytest

from rankone.binary_membership import binary_membership_check
from rankone.field import Field
from rankone.fibre import fibre_check, verify_fibre
from rankone.operations import execute


E11=[[1,0],[0,0]]
E22=[[0,0],[0,1]]
J=[[1,1],[1,1]]
CUSTOM9={"order":9,"irreducible_poly":[1,0,1]}


@pytest.mark.parametrize("q",[2,3,4,5])
def test_public_transposed_two_row_constructor_preserves_declared_shape(q):
    output=execute({"operation":"construct","q":q,"rows":3,"columns":2,
                    "construction":"two_row","k":2})
    assert output["status"]=="VALID"
    problem=output["problem"]
    assert (problem["rows"],problem["columns"])==(3,2)
    assert len(problem["generators"][0])==3
    assert len(problem["generators"][0][0])==2
    assert verify_fibre(fibre_check(problem))["verified"] is True


@pytest.mark.parametrize("operation",["bounds","table","construct","search","known_classes"])
def test_operation_metadata_is_validated_before_model_canonicalization(operation):
    for key,value in (("characteristic",9),("degree",1),("encoding","integer residues mod nine")):
        request={"operation":operation,"field":{**CUSTOM9,key:value},"rows":2,"columns":3}
        with pytest.raises(ValueError):
            execute(request)


@pytest.mark.parametrize("operation",["bounds","table","known_classes"])
def test_selected_nondefault_extension_model_survives_research_views(operation):
    requested=Field(9,[1,0,1]).metadata()
    output=execute({"operation":operation,"field":CUSTOM9,"rows":2,"columns":3,"k":2})
    report=output["report"]
    if operation=="table":
        for entry in report["entries"]:
            assert entry["field"]==requested
            assert entry["witness"]["field"]==requested
    elif operation=="known_classes":
        assert report["class_count"]==1
        assert report["representatives"][0]["field"]==requested
    else:
        assert report["field"]==requested
        assert report["witness"]["field"]==requested


def test_saved_legacy_binary_certificates_can_be_replayed_and_tampering_rejected():
    report=binary_membership_check({"field":{"order":2},"generators":[E11,E22,J]})
    assert report["status"]=="VALID"
    assert execute({"operation":"replay","data":report})["status"]=="VERIFIED"
    wrapped={"operation":"check","report":report,"status":"VALID"}
    assert execute({"operation":"replay","data":wrapped})["status"]=="VERIFIED"
    damaged=deepcopy(report)
    damaged["rank_one_members"].pop()
    assert execute({"operation":"replay","data":damaged})["status"]=="REJECTED"


def test_zero_basis_public_paths_remain_meaningful():
    instance={"field":{"order":2},"rows":2,"columns":3,"generators":[]}
    for method in ("fibre","hybrid","normalized","reference","binary_ambient"):
        out=execute({"operation":"check","method":method,"problem":instance})
        assert out["status"]=="VALID",method
    out=execute({"operation":"distribution","problem":instance})
    assert out["status"]=="COMPLETE"
    assert out["report"]["matrix_counts"]=={"0":1}


def test_constructor_prediction_never_upgrades_an_incomplete_check():
    output=execute({"operation":"construct","q":3,"rows":3,"columns":3,
                    "construction":"verified","max_seconds":0})
    assert output["status"]=="INCONCLUSIVE"
    assert output["check"]["status"]=="INCONCLUSIVE"
    assert output["construction_status"]=="VALID"


def test_general_field_search_limits_and_checked_bound_path_have_distinct_meanings():
    out=execute({"operation":"search","q":3,"rows":3,"columns":3,
                 "max_candidates":0,"max_nodes":0,"use_known":False})
    assert out["status"]=="INCONCLUSIVE"
    assert not out["report"]["maximum_proved"]
    assert out["report"]["lower_bound"]==0
    exact=execute({"operation":"search","q":3,"rows":3,"columns":3,
                   "max_candidates":0,"max_nodes":0})
    assert exact["status"]=="EXACT"
    assert exact["report"]["upper_bound_proof"]
    assert exact["report"]["witness_replay"]["verified"] is True


@pytest.mark.parametrize("method_mutation",["unknown","missing"])
def test_saved_checker_cannot_bypass_certificate_replay_by_disguising_its_method(method_mutation):
    from rankone.certificates import verify_saved
    instance={"field":{"order":3},"generators":[E11,E22]}
    report=fibre_check(instance)
    report["certificate"]["rank_one_directions"]=999
    if method_mutation=="unknown":
        report["method"]="unknown_method"
    else:
        del report["method"]
    assert verify_saved(report)["verified"] is False


def test_complete_collection_heading_stays_within_the_supplied_list():
    from rankone.equivalence import classify_families
    from rankone.presentation import research_explanation
    instance={"field":{"order":2},"generators":[E11,E22]}
    report=classify_families([instance],domain_complete=True)
    bundle={"operation":"classify","status":report["status"],"report":report}
    text=research_explanation(bundle)
    assert "supplied" in text.lower()
    assert "does not exhaust all possible" in text.lower()
    assert "Complete classification for the stated field" not in text


@pytest.mark.parametrize("method",["fibre","hybrid","normalized","reference","binary_ambient"])
def test_public_zero_wall_time_never_starts_an_unbounded_legacy_check(method):
    out=execute({"operation":"check","method":method,
                 "problem":{"field":{"order":2},"generators":[E11,E22]},
                 "max_seconds":0})
    assert out["status"]=="INCONCLUSIVE"
