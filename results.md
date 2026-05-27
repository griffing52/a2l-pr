Policy	Success Rate	Mean Return	Std Return	Notes
BC	44%	0.44	0.496	baseline BC
BC + gated residual	62%	0.62	0.485	residual gate never crossed threshold
Diffusion	84%	0.84	0.367	baseline diffusion
Diffusion + gated residual	92%	0.92	0.271	residual gate never crossed threshold
Important caveat: for both residual variants, mean_interventions=0.0. So this run shows the gated residual is no longer destroying clean policy behavior, but it also means the residual was effectively inactive at the conservative threshold:

gate_threshold=0.7
BC+residual max gate: 0.579
Diffusion+residual max gate: 0.571
So the apparent improvement is likely rollout stochasticity / seed trajectory variation, not actual residual correction.