import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from enum import Enum

class TerminationLogic(Enum):
    """Termination predicate logic."""
    ANY = "ANY"  # Terminate if ANY predicate is true
    ALL = "ALL"  # Terminate if ALL predicates are true

@dataclass
class PredicateSpec:
    """A single termination predicate."""
    lhs: str  # left-hand side (metric name)
    cmp: str  # comparison operator: >=, <=, ==, >, <
    rhs: float  # right-hand side (threshold value)
    
    def to_dict(self):
        return asdict(self)

@dataclass
class StepSpec:
    """A single step in the recovery FSM."""
    id: str  # step identifier (e.g., "S1")
    call: str  # function call as string
    termination: Dict[str, Any]  # termination spec with logic, predicates, max_steps
    next_on_success: str  # next state on success (step id or "HALT")
    next_on_timeout: str  # next state on timeout
    
    def to_dict(self):
        return asdict(self)

class RecoveryCodeGenerator:
    """Generate deterministic recovery code in DROID-100 DSL format."""
    
    @staticmethod
    def create_predicate(
        lhs: str, cmp: str, rhs: float
    ) -> PredicateSpec:
        """Create a termination predicate."""
        return PredicateSpec(lhs=lhs, cmp=cmp, rhs=rhs)
    
    @staticmethod
    def create_termination_spec(
        logic: TerminationLogic,
        predicates: List[PredicateSpec],
        max_steps: int
    ) -> Dict[str, Any]:
        """Create termination specification."""
        return {
            "logic": logic.value,
            "predicates": [p.to_dict() for p in predicates],
            "max_steps": max_steps
        }
    
    @staticmethod
    def create_step(
        step_id: str,
        function_call: str,
        termination: Dict[str, Any],
        next_on_success: str = "HALT",
        next_on_timeout: str = "HALT"
    ) -> StepSpec:
        """Create a recovery step."""
        return StepSpec(
            id=step_id,
            call=function_call,
            termination=termination,
            next_on_success=next_on_success,
            next_on_timeout=next_on_timeout
        )
    
    @staticmethod
    def generate_underreach_recovery(params: Dict[str, float]) -> Dict[str, Any]:
        """Recovery for under-reach idle: forward nudge + resume."""
        steps = [
            RecoveryCodeGenerator.create_step(
                step_id="S1",
                function_call="estimate_local_forward_axis(source=\"end_effector_orientation\", fallback=\"recent_motion_direction\", window_steps=6)",
                termination=RecoveryCodeGenerator.create_termination_spec(
                    TerminationLogic.ANY,
                    [RecoveryCodeGenerator.create_predicate("elapsed_steps", ">=", 1)],
                    1
                ),
                next_on_success="S2",
                next_on_timeout="S2"
            ),
            RecoveryCodeGenerator.create_step(
                step_id="S2",
                function_call=f"move_relative_ee(direction=\"forward\", distance_m={params['forward_nudge_distance_m']:.4f}, speed_scale={params['speed_scale']:.2f})",
                termination=RecoveryCodeGenerator.create_termination_spec(
                    TerminationLogic.ANY,
                    [
                        RecoveryCodeGenerator.create_predicate("relative_progress_m", ">=", params['forward_nudge_distance_m']),
                        RecoveryCodeGenerator.create_predicate("contact_confidence", ">=", 0.70)
                    ],
                    16
                ),
                next_on_success="S3",
                next_on_timeout="S3"
            ),
            RecoveryCodeGenerator.create_step(
                step_id="S3",
                function_call="resume_reference_trajectory(blend_mode=\"smooth\")",
                termination=RecoveryCodeGenerator.create_termination_spec(
                    TerminationLogic.ANY,
                    [RecoveryCodeGenerator.create_predicate("elapsed_steps", ">=", 1)],
                    1
                ),
                next_on_success="HALT",
                next_on_timeout="HALT"
            )
        ]
        
        return {
            "dsl_version": "deterministic_recovery_fsm_v1",
            "execution_model": "sequential_fsm",
            "severity_hint": "moderate",
            "parameters": params,
            "robot_interface_contract": {
                "requires_live_ee_pose": True,
                "requires_gripper_state_if_used": False,
                "requires_metric_evaluator": True,
                "history_context_required": False
            },
            "program": [s.to_dict() for s in steps]
        }
    
    @staticmethod
    def generate_premature_close_recovery(params: Dict[str, float]) -> Dict[str, Any]:
        """Recovery for premature close: open + align + close + resume."""
        steps = [
            RecoveryCodeGenerator.create_step(
                step_id="S1",
                function_call="set_gripper(value=\"open\")",
                termination=RecoveryCodeGenerator.create_termination_spec(
                    TerminationLogic.ANY,
                    [RecoveryCodeGenerator.create_predicate("gripper_opened", ">=", 0.9)],
                    5
                ),
                next_on_success="S2",
                next_on_timeout="S2"
            ),
            RecoveryCodeGenerator.create_step(
                step_id="S2",
                function_call=f"align_end_effector_to_object(mode=\"centroid_or_grasp_pose\", xy_tolerance_m={params['xy_tolerance_m']:.4f}, speed_scale=0.35)",
                termination=RecoveryCodeGenerator.create_termination_spec(
                    TerminationLogic.ANY,
                    [RecoveryCodeGenerator.create_predicate("alignment_error_m", "<=", params['xy_tolerance_m'])],
                    int(params['max_align_steps'])
                ),
                next_on_success="S3",
                next_on_timeout="S3"
            ),
            RecoveryCodeGenerator.create_step(
                step_id="S3",
                function_call="set_gripper(value=\"close\")",
                termination=RecoveryCodeGenerator.create_termination_spec(
                    TerminationLogic.ANY,
                    [RecoveryCodeGenerator.create_predicate("gripper_closed", ">=", 0.8)],
                    5
                ),
                next_on_success="S4",
                next_on_timeout="S4"
            ),
            RecoveryCodeGenerator.create_step(
                step_id="S4",
                function_call="resume_reference_trajectory(blend_mode=\"smooth\")",
                termination=RecoveryCodeGenerator.create_termination_spec(
                    TerminationLogic.ANY,
                    [RecoveryCodeGenerator.create_predicate("elapsed_steps", ">=", 1)],
                    1
                ),
                next_on_success="HALT",
                next_on_timeout="HALT"
            )
        ]
        
        return {
            "dsl_version": "deterministic_recovery_fsm_v1",
            "execution_model": "sequential_fsm",
            "severity_hint": "moderate",
            "parameters": params,
            "robot_interface_contract": {
                "requires_live_ee_pose": True,
                "requires_gripper_state_if_used": True,
                "requires_metric_evaluator": True,
                "history_context_required": False
            },
            "program": [s.to_dict() for s in steps]
        }
    
    @staticmethod
    def generate_premature_open_recovery(params: Dict[str, float]) -> Dict[str, Any]:
        """Recovery for premature open: re-close + lift + resume."""
        steps = [
            RecoveryCodeGenerator.create_step(
                step_id="S1",
                function_call="set_gripper(value=\"close\")",
                termination=RecoveryCodeGenerator.create_termination_spec(
                    TerminationLogic.ANY,
                    [RecoveryCodeGenerator.create_predicate("gripper_closed", ">=", 0.8)],
                    5
                ),
                next_on_success="S2",
                next_on_timeout="S2"
            ),
            RecoveryCodeGenerator.create_step(
                step_id="S2",
                function_call=f"move_relative_world(axis=\"z\", distance_m={params['lift_distance_m']:.4f}, speed_scale={params['speed_scale']:.2f})",
                termination=RecoveryCodeGenerator.create_termination_spec(
                    TerminationLogic.ANY,
                    [RecoveryCodeGenerator.create_predicate("relative_progress_m", ">=", params['lift_distance_m'])],
                    10
                ),
                next_on_success="S3",
                next_on_timeout="S3"
            ),
            RecoveryCodeGenerator.create_step(
                step_id="S3",
                function_call="resume_reference_trajectory(blend_mode=\"smooth\")",
                termination=RecoveryCodeGenerator.create_termination_spec(
                    TerminationLogic.ANY,
                    [RecoveryCodeGenerator.create_predicate("elapsed_steps", ">=", 1)],
                    1
                ),
                next_on_success="HALT",
                next_on_timeout="HALT"
            )
        ]
        
        return {
            "dsl_version": "deterministic_recovery_fsm_v1",
            "execution_model": "sequential_fsm",
            "severity_hint": "moderate",
            "parameters": params,
            "robot_interface_contract": {
                "requires_live_ee_pose": True,
                "requires_gripper_state_if_used": True,
                "requires_metric_evaluator": True,
                "history_context_required": False
            },
            "program": [s.to_dict() for s in steps]
        }
    
    @staticmethod
    def generate_lateral_drift_recovery(params: Dict[str, float]) -> Dict[str, Any]:
        """Recovery for lateral drift: re-center + resume."""
        steps = [
            RecoveryCodeGenerator.create_step(
                step_id="S1",
                function_call=f"align_lateral_error(reference=\"target_or_reference_path\", frame=\"task_plane_xy\", xy_tolerance_m={params['xy_tolerance_m']:.4f}, speed_scale={params['speed_scale']:.2f})",
                termination=RecoveryCodeGenerator.create_termination_spec(
                    TerminationLogic.ANY,
                    [RecoveryCodeGenerator.create_predicate("alignment_error_m", "<=", params['xy_tolerance_m'])],
                    20
                ),
                next_on_success="S2",
                next_on_timeout="S2"
            ),
            RecoveryCodeGenerator.create_step(
                step_id="S2",
                function_call="resume_reference_trajectory(blend_mode=\"smooth\")",
                termination=RecoveryCodeGenerator.create_termination_spec(
                    TerminationLogic.ANY,
                    [RecoveryCodeGenerator.create_predicate("elapsed_steps", ">=", 1)],
                    1
                ),
                next_on_success="HALT",
                next_on_timeout="HALT"
            )
        ]
        
        return {
            "dsl_version": "deterministic_recovery_fsm_v1",
            "execution_model": "sequential_fsm",
            "severity_hint": "moderate",
            "parameters": params,
            "robot_interface_contract": {
                "requires_live_ee_pose": True,
                "requires_gripper_state_if_used": False,
                "requires_metric_evaluator": True,
                "history_context_required": False
            },
            "program": [s.to_dict() for s in steps]
        }
    
    @staticmethod
    def generate_recovery_code(perturbation_result) -> Dict[str, Any]:
        """Generate recovery code for a perturbation result."""
        from a2l_pr.perturbations.generator import PerturbationType
        
        params = perturbation_result.parameters
        perturb_type = perturbation_result.perturbation_type
        
        if perturb_type == PerturbationType.UNDERREACH_IDLE:
            return RecoveryCodeGenerator.generate_underreach_recovery(params)
        elif perturb_type == PerturbationType.PREMATURE_CLOSE:
            return RecoveryCodeGenerator.generate_premature_close_recovery(params)
        elif perturb_type == PerturbationType.PREMATURE_OPEN:
            return RecoveryCodeGenerator.generate_premature_open_recovery(params)
        elif perturb_type == PerturbationType.LATERAL_DRIFT:
            return RecoveryCodeGenerator.generate_lateral_drift_recovery(params)
        else:
            return {
                "dsl_version": "deterministic_recovery_fsm_v1",
                "execution_model": "sequential_fsm",
                "parameters": params,
                "program": []
            }

def recovery_to_json(recovery_code: Dict[str, Any]) -> str:
    """Convert recovery code to JSON string."""
    return json.dumps(recovery_code, indent=2)

def recovery_from_json(json_str: str) -> Dict[str, Any]:
    """Parse recovery code from JSON string."""
    return json.loads(json_str)
