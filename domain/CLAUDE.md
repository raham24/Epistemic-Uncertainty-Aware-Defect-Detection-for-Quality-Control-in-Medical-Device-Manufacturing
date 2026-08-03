# Domain Spec Contract

Single YAML file is the contract between every component. The generator reads it. The ontology mirrors it. Evaluation references it. Schema:

```yaml
device_class: external_insulin_pump
standards:
  - IEC 60601-2-24
  - AAMI/ANSI ID26
defects:
  - name: no_failure
    prior: 0.85  # calibrated from MAUDE
  - name: occlusion_event
    prior: 0.07
  # ...
mechanisms:
  stage_1_delivery:
    - name: pump_motor_stall
    - name: tubing_kink
    - name: no_mechanism
  stage_2_dose_accuracy:
    - name: stepper_drift
    - name: no_mechanism
parameters:
  - id: motor_current
    nominal: 120.0
    usl: 180.0
    lsl: 60.0
    unit: mA
    stage: stage_1_delivery
  - id: occlusion_pressure
    nominal: 100.0
    usl: 250.0
    unit: mmHg
    stage: stage_1_delivery
  # ... 4-6 parameters total
causal_edges:
  - from: occlusion_event
    via: pump_motor_stall
    parameter: motor_current
    direction: high
  # ...
correlations:
  motor_current: { occlusion_pressure: 0.7, battery_voltage: -0.3 }
  # ...
risk_function:
  p_L: 0.05
  p_M: 0.70
  p_H: 0.99
  kappa: 0.015
generator:
  n_records: 200000
  n_batches: 100
  records_per_batch: 2000
  seed: 42
```

When in doubt about generator behavior, the YAML is authoritative.
