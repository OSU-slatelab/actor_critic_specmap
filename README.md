# Spectral Mapping Using Mimic Loss for Robust ASR

This repository implements "Spectral Mapping Using Mimic Loss for Robust Speech Recognition"
(ICASSP 2018) by Deblin Bagchi, Peter Plantinga, Adam Stiff, and Eric Fosler-Lussier.

When we started this project, we called the spectral mapper an "actor" and the spectral
classifier a "critic", but changed the terminology when writing the paper.

Mimic loss is a concept we introduce in this paper that falls under the general
umbrella of knowledge distillation (like student-teacher learning). However, instead
of training two models to do the same task, mimic loss trains a pre-processor
model to do a different task from the teacher model. It does this by freezing the
weights of the teacher model and backpropagating errors through it to the student model.
More details available in the paper.
