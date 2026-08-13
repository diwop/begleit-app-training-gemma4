# ADR-003 Environments for training and inference

Containers for Axolotl (training) and vLLM (inference) are run using apptainer on the HSUper GPU nodes.
The actual code for training and inferences resides just in the repository and is not baked into docker or apptainer images.

## Rationale

The pipeline is strongly based on the heavy dependencies Axolotl and vLLM being launched by rather simple Python scripts that change fast.
Expecially Axolotl is extremely error-prone to compile.
HSUper cannot even run docker images directly but requires apptainer images which would increase cycle times further.
Axolotl and vLLM are hardly changed while the scripts for training and inference are subject to faster changes.
Therefore the considered approach of maintaining a Docker or even Apptainer image for training and/or inference is discarded.
The training code is run in an Axolotl container while the code for inference is run in an vLLM container.