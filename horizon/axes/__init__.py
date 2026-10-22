from . import coding, icl_language, long_context, math_reasoning, planning, rule_induction, world_model

# Calibration is not listed: it is derived from the confidence every model
# reports on every item of every axis (see scoring.calibration).
AXES = {m.AXIS: m.generate for m in (rule_induction, math_reasoning, coding, planning,
                                      icl_language, long_context, world_model)}
