# ood-class-separation

This repo contains additional implementation details, numerical results and the Python code for the paper "Class Separation is not what you need for Relational Reasonig-based OOD Detection",
accepted at ICIAP 2023.

## Implementation details

We mainly follow the architectural implementation presented in the original ReSeND work
([ECCV, 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850181.pdf)),
with the only exception of the final *semantic similarity head* $c_\delta$ which we implement as a 2-layer MLP rather than a single FC.
As for the training procedure, we leverage ImageNet-1K to build random pairs of images by associating an
anchor sample to both a different image from the same category and multiple others from different classes.
All the images undergo a sequence of random transformations consisting of cropping, horizontal flip, color jitter and grayscale.
We train for 33k steps with a batch size of 4096 image pairs and a learning rate set to 0.008 (after a linear warmup
for the first 500 steps), using the [LARS](https://arxiv.org/abs/1708.03888) optimizer with a weight decay $5 \cdot 10^{-4}$ and momentum 0.9.

In order to compute the $R^2$ values, we follow the same random sampling strategy used in the training phase to build
12k image pairs which are then fed as input to the model, in order to obtain the corresponding relational representations from the penultimate
layer.


## Results with different training objectives

We report the numerical results obtained by ReSeND when trained with the training objectives presented in our work. <br>
With regard to the focal loss, we adopt $\gamma = 3$, based on the results of Mukhoti et al.
([NeurIPS, 2020](https://proceedings.neurips.cc/paper/2020/file/aeb7b30ef1d024a76f21a1d40e30c302-Paper.pdf)). 

### Intra-Domain

<div class="adjustbox">
<div id="tab:losses_results">
<table>
<tbody>
<tr class="odd">
<td rowspan="2" style="text-align: center;"><strong>Loss</strong></td>
<td rowspan="2" style="text-align: center;"><strong>Param</strong></td>
<td colspan="2"
style="text-align: center;"><strong>Texture</strong></td>
<td colspan="2" style="text-align: center;"><strong>Real</strong></td>
<td colspan="2" style="text-align: center;"><strong>Sketch</strong></td>
<td colspan="2"
style="text-align: center;"><strong>Painting</strong></td>
<td colspan="2" style="text-align: center;"><strong>Avg</strong></td>
<td style="text-align: center;"><strong>ImageNet</strong></td>
</tr>
<tr class="even">
<td style="text-align: center;">AUC <span
class="math inline">↑</span></td>
<td style="text-align: center;">FPR <span
class="math inline">↓</span></td>
<td style="text-align: center;">AUC <span
class="math inline">↑</span></td>
<td style="text-align: center;">FPR <span
class="math inline">↓</span></td>
<td style="text-align: center;">AUC <span
class="math inline">↑</span></td>
<td style="text-align: center;">FPR <span
class="math inline">↓</span></td>
<td style="text-align: center;">AUC <span
class="math inline">↑</span></td>
<td style="text-align: center;">FPR <span
class="math inline">↓</span></td>
<td style="text-align: center;">AUC <span
class="math inline">↑</span></td>
<td style="text-align: center;">FPR <span
class="math inline">↓</span></td>
<td style="text-align: center;"><span
class="math inline"><em>R</em><sup>2</sup></span></td>
</tr>
<tr class="odd">
<td rowspan="3" style="text-align: center;">MSE</td>
<td style="text-align: center;">c=1</td>
<td style="text-align: center;">0.669</td>
<td style="text-align: center;">0.883</td>
<td style="text-align: center;">0.704</td>
<td style="text-align: center;">0.801</td>
<td style="text-align: center;">0.588</td>
<td style="text-align: center;">0.934</td>
<td style="text-align: center;">0.680</td>
<td style="text-align: center;">0.853</td>
<td style="text-align: center;">0.660</td>
<td style="text-align: center;">0.868</td>
<td style="text-align: center;">0.520</td>
</tr>
<tr class="even">
<td style="text-align: center;">c=10</td>
<td style="text-align: center;">0.686</td>
<td style="text-align: center;"><strong>0.881</strong></td>
<td style="text-align: center;">0.776</td>
<td style="text-align: center;">0.771</td>
<td style="text-align: center;">0.615</td>
<td style="text-align: center;">0.922</td>
<td style="text-align: center;"><strong>0.729</strong></td>
<td style="text-align: center;"><strong>0.809</strong></td>
<td style="text-align: center;">0.702</td>
<td style="text-align: center;">0.846</td>
<td style="text-align: center;">0.111</td>
</tr>
<tr class="odd">
<td style="text-align: center;">c=50</td>
<td style="text-align: center;">0.675</td>
<td style="text-align: center;">0.897</td>
<td style="text-align: center;">0.777</td>
<td style="text-align: center;">0.769</td>
<td style="text-align: center;">0.610</td>
<td style="text-align: center;">0.910</td>
<td style="text-align: center;">0.719</td>
<td style="text-align: center;">0.841</td>
<td style="text-align: center;">0.695</td>
<td style="text-align: center;">0.854</td>
<td style="text-align: center;">0.015</td>
</tr>
<tr class="even">
<td rowspan="2" style="text-align: center;">BCE</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">0.663</td>
<td style="text-align: center;">0.897</td>
<td style="text-align: center;">0.654</td>
<td style="text-align: center;">0.866</td>
<td style="text-align: center;">0.569</td>
<td style="text-align: center;">0.939</td>
<td style="text-align: center;">0.683</td>
<td style="text-align: center;">0.864</td>
<td style="text-align: center;">0.642</td>
<td style="text-align: center;">0.892</td>
<td style="text-align: center;">0.592</td>
</tr>
<tr class="odd">
<td style="text-align: center;">focal</td>
<td style="text-align: center;">0.645</td>
<td style="text-align: center;">0.926</td>
<td style="text-align: center;">0.706</td>
<td style="text-align: center;">0.805</td>
<td style="text-align: center;">0.560</td>
<td style="text-align: center;">0.945</td>
<td style="text-align: center;">0.696</td>
<td style="text-align: center;">0.829</td>
<td style="text-align: center;">0.652</td>
<td style="text-align: center;">0.876</td>
<td style="text-align: center;">0.441</td>
</tr>
<tr class="even">
<td rowspan="2" style="text-align: center;">CE</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">0.614</td>
<td style="text-align: center;">0.948</td>
<td style="text-align: center;">0.651</td>
<td style="text-align: center;">0.847</td>
<td style="text-align: center;">0.554</td>
<td style="text-align: center;">0.953</td>
<td style="text-align: center;">0.648</td>
<td style="text-align: center;">0.897</td>
<td style="text-align: center;">0.617</td>
<td style="text-align: center;">0.911</td>
<td style="text-align: center;">0.592</td>
</tr>
<tr class="odd">
<td style="text-align: center;">focal</td>
<td style="text-align: center;">0.674</td>
<td style="text-align: center;">0.885</td>
<td style="text-align: center;">0.742</td>
<td style="text-align: center;">0.780</td>
<td style="text-align: center;">0.590</td>
<td style="text-align: center;">0.932</td>
<td style="text-align: center;">0.713</td>
<td style="text-align: center;">0.856</td>
<td style="text-align: center;">0.680</td>
<td style="text-align: center;">0.863</td>
<td style="text-align: center;">0.362</td>
</tr>
<tr class="even">
<td rowspan="3" style="text-align: center;">H</td>
<td style="text-align: center;">&delta;=1</td>
<td style="text-align: center;">0.620</td>
<td style="text-align: center;">0.924</td>
<td style="text-align: center;">0.702</td>
<td style="text-align: center;"><strong>0.730</strong></td>
<td style="text-align: center;">0.519</td>
<td style="text-align: center;">0.946</td>
<td style="text-align: center;">0.609</td>
<td style="text-align: center;">0.869</td>
<td style="text-align: center;">0.612</td>
<td style="text-align: center;">0.867</td>
<td style="text-align: center;">0.487</td>
</tr>
<tr class="odd">
<td style="text-align: center;">&delta;=0.1</td>
<td style="text-align: center;">0.659</td>
<td style="text-align: center;">0.911</td>
<td style="text-align: center;">0.784</td>
<td style="text-align: center;">0.763</td>
<td style="text-align: center;">0.569</td>
<td style="text-align: center;">0.937</td>
<td style="text-align: center;">0.697</td>
<td style="text-align: center;">0.862</td>
<td style="text-align: center;">0.677</td>
<td style="text-align: center;">0.868</td>
<td style="text-align: center;">0.251</td>
</tr>
<tr class="even">
<td style="text-align: center;">&delta;=0.01</td>
<td style="text-align: center;"><strong>0.688</strong></td>
<td style="text-align: center;">0.885</td>
<td style="text-align: center;"><strong>0.798</strong></td>
<td style="text-align: center;">0.755</td>
<td style="text-align: center;"><strong>0.638</strong></td>
<td style="text-align: center;"><strong>0.898</strong></td>
<td style="text-align: center;">0.719</td>
<td style="text-align: center;">0.826</td>
<td style="text-align: center;"><strong>0.711</strong></td>
<td style="text-align: center;"><strong>0.841</strong></td>
<td style="text-align: center;">0.101</td>
</tr>
</tbody>
</table>
</div>
</div>

### Cross-Domain

<div class="adjustbox">
<div id="tab:losses_results">
<table>
<tbody>
<tr class="odd">
<td rowspan="2" style="text-align: center;"><strong>Loss</strong></td>
<td rowspan="2" style="text-align: center;"><strong>Param</strong></td>
<td colspan="2"
style="text-align: center;"><strong>Real-Paint.</strong></td>
<td colspan="2"
style="text-align: center;"><strong>Real-Sketch</strong></td>
<td colspan="2"
style="text-align: center;"><strong>Paint.-Real</strong></td>
<td colspan="2"
style="text-align: center;"><strong>Paint.-Sketch</strong></td>
<td colspan="2"
style="text-align: center;"><strong>Sketch-Real</strong></td>
<td colspan="2"
style="text-align: center;"><strong>Sketch-Paint.</strong></td>
<td colspan="2" style="text-align: center;"><strong>Avg</strong></td>
<td style="text-align: center;"><strong>ImageNet</strong></td>
</tr>
<tr class="even">
<td style="text-align: center;">AUC <span
class="math inline">↑</span></td>
<td style="text-align: center;">FPR <span
class="math inline">↓</span></td>
<td style="text-align: center;">AUC <span
class="math inline">↑</span></td>
<td style="text-align: center;">FPR <span
class="math inline">↓</span></td>
<td style="text-align: center;">AUC <span
class="math inline">↑</span></td>
<td style="text-align: center;">FPR <span
class="math inline">↓</span></td>
<td style="text-align: center;">AUC <span
class="math inline">↑</span></td>
<td style="text-align: center;">FPR <span
class="math inline">↓</span></td>
<td style="text-align: center;">AUC <span
class="math inline">↑</span></td>
<td style="text-align: center;">FPR <span
class="math inline">↓</span></td>
<td style="text-align: center;">AUC <span
class="math inline">↑</span></td>
<td style="text-align: center;">FPR <span
class="math inline">↓</span></td>
<td style="text-align: center;">AUC <span
class="math inline">↑</span></td>
<td style="text-align: center;">FPR <span
class="math inline">↓</span></td>
<td style="text-align: center;"><span
class="math inline"><em>R</em><sup>2</sup></span></td>
</tr>
<tr class="odd">
<td rowspan="3" style="text-align: center;">MSE</td>
<td style="text-align: center;">c=1</td>
<td style="text-align: center;">0.596</td>
<td style="text-align: center;">0.931</td>
<td style="text-align: center;">0.564</td>
<td style="text-align: center;">0.929</td>
<td style="text-align: center;">0.707</td>
<td style="text-align: center;">0.869</td>
<td style="text-align: center;">0.563</td>
<td style="text-align: center;">0.933</td>
<td style="text-align: center;">0.607</td>
<td style="text-align: center;">0.925</td>
<td style="text-align: center;">0.569</td>
<td style="text-align: center;">0.943</td>
<td style="text-align: center;">0.601</td>
<td style="text-align: center;">0.922</td>
<td style="text-align: center;">0.520</td>
</tr>
<tr class="even">
<td style="text-align: center;">c=10</td>
<td style="text-align: center;"><strong>0.661</strong></td>
<td style="text-align: center;">0.903</td>
<td style="text-align: center;">0.580</td>
<td style="text-align: center;">0.926</td>
<td style="text-align: center;"><strong>0.728</strong></td>
<td style="text-align: center;">0.891</td>
<td style="text-align: center;">0.575</td>
<td style="text-align: center;">0.928</td>
<td style="text-align: center;">0.679</td>
<td style="text-align: center;">0.903</td>
<td style="text-align: center;">0.645</td>
<td style="text-align: center;">0.918</td>
<td style="text-align: center;"><strong>0.645</strong></td>
<td style="text-align: center;">0.911</td>
<td style="text-align: center;">0.111</td>
</tr>
<tr class="odd">
<td style="text-align: center;">c=50</td>
<td style="text-align: center;">0.656</td>
<td style="text-align: center;">0.920</td>
<td style="text-align: center;">0.558</td>
<td style="text-align: center;">0.940</td>
<td style="text-align: center;">0.726</td>
<td style="text-align: center;">0.874</td>
<td style="text-align: center;">0.560</td>
<td style="text-align: center;">0.926</td>
<td style="text-align: center;"><strong>0.691</strong></td>
<td style="text-align: center;"><strong>0.875</strong></td>
<td style="text-align: center;">0.642</td>
<td style="text-align: center;"><strong>0.900</strong></td>
<td style="text-align: center;">0.639</td>
<td style="text-align: center;">0.906</td>
<td style="text-align: center;">0.015</td>
</tr>
<tr class="even">
<td rowspan="2" style="text-align: center;">BCE</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">0.608</td>
<td style="text-align: center;">0.931</td>
<td style="text-align: center;">0.556</td>
<td style="text-align: center;">0.931</td>
<td style="text-align: center;">0.673</td>
<td style="text-align: center;">0.896</td>
<td style="text-align: center;">0.582</td>
<td style="text-align: center;">0.929</td>
<td style="text-align: center;">0.660</td>
<td style="text-align: center;">0.896</td>
<td style="text-align: center;">0.652</td>
<td style="text-align: center;">0.921</td>
<td style="text-align: center;">0.622</td>
<td style="text-align: center;">0.917</td>
<td style="text-align: center;">0.592</td>
</tr>
<tr class="odd">
<td style="text-align: center;">focal</td>
<td style="text-align: center;">0.648</td>
<td style="text-align: center;"><strong>0.901</strong></td>
<td style="text-align: center;">0.565</td>
<td style="text-align: center;">0.937</td>
<td style="text-align: center;">0.697</td>
<td style="text-align: center;">0.898</td>
<td style="text-align: center;">0.561</td>
<td style="text-align: center;">0.943</td>
<td style="text-align: center;">0.686</td>
<td style="text-align: center;">0.881</td>
<td style="text-align: center;">0.642</td>
<td style="text-align: center;">0.916</td>
<td style="text-align: center;">0.633</td>
<td style="text-align: center;">0.913</td>
<td style="text-align: center;">0.441</td>
</tr>
<tr class="even">
<td rowspan="2" style="text-align: center;">CE</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">0.603</td>
<td style="text-align: center;">0.916</td>
<td style="text-align: center;">0.554</td>
<td style="text-align: center;">0.942</td>
<td style="text-align: center;">0.664</td>
<td style="text-align: center;">0.896</td>
<td style="text-align: center;">0.566</td>
<td style="text-align: center;">0.935</td>
<td style="text-align: center;">0.669</td>
<td style="text-align: center;">0.912</td>
<td style="text-align: center;">0.645</td>
<td style="text-align: center;">0.934</td>
<td style="text-align: center;">0.617</td>
<td style="text-align: center;">0.923</td>
<td style="text-align: center;">0.592</td>
</tr>
<tr class="odd">
<td style="text-align: center;">focal</td>
<td style="text-align: center;">0.649</td>
<td style="text-align: center;">0.931</td>
<td style="text-align: center;">0.572</td>
<td style="text-align: center;">0.935</td>
<td style="text-align: center;">0.711</td>
<td style="text-align: center;">0.907</td>
<td style="text-align: center;">0.566</td>
<td style="text-align: center;">0.935</td>
<td style="text-align: center;">0.679</td>
<td style="text-align: center;">0.903</td>
<td style="text-align: center;"><strong>0.659</strong></td>
<td style="text-align: center;">0.923</td>
<td style="text-align: center;">0.639</td>
<td style="text-align: center;">0.922</td>
<td style="text-align: center;">0.362</td>
</tr>
<tr class="even">
<td rowspan="3" style="text-align: center;">H</td>
<td style="text-align: center;">&delta;=1</td>
<td style="text-align: center;">0.587</td>
<td style="text-align: center;">0.922</td>
<td style="text-align: center;">0.552</td>
<td style="text-align: center;">0.933</td>
<td style="text-align: center;">0.677</td>
<td style="text-align: center;">0.879</td>
<td style="text-align: center;">0.541</td>
<td style="text-align: center;">0.917</td>
<td style="text-align: center;">0.558</td>
<td style="text-align: center;">0.894</td>
<td style="text-align: center;">0.525</td>
<td style="text-align: center;">0.935</td>
<td style="text-align: center;">0.573</td>
<td style="text-align: center;">0.913</td>
<td style="text-align: center;">0.487</td>
</tr>
<tr class="odd">
<td style="text-align: center;">&delta;=0.1</td>
<td style="text-align: center;">0.633</td>
<td style="text-align: center;">0.939</td>
<td style="text-align: center;">0.554</td>
<td style="text-align: center;">0.947</td>
<td style="text-align: center;">0.705</td>
<td style="text-align: center;">0.889</td>
<td style="text-align: center;">0.549</td>
<td style="text-align: center;">0.936</td>
<td style="text-align: center;">0.540</td>
<td style="text-align: center;">0.927</td>
<td style="text-align: center;">0.519</td>
<td style="text-align: center;">0.933</td>
<td style="text-align: center;">0.583</td>
<td style="text-align: center;">0.928</td>
<td style="text-align: center;">0.251</td>
</tr>
<tr class="even">
<td style="text-align: center;">&delta;=0.01</td>
<td style="text-align: center;">0.639</td>
<td style="text-align: center;">0.938</td>
<td style="text-align: center;"><strong>0.583</strong></td>
<td style="text-align: center;"><strong>0.919</strong></td>
<td style="text-align: center;">0.720</td>
<td style="text-align: center;"><strong>0.864</strong></td>
<td style="text-align: center;"><strong>0.590</strong></td>
<td style="text-align: center;"><strong>0.899</strong></td>
<td style="text-align: center;">0.679</td>
<td style="text-align: center;">0.895</td>
<td style="text-align: center;">0.637</td>
<td style="text-align: center;">0.914</td>
<td style="text-align: center;">0.641</td>
<td style="text-align: center;"><strong>0.905</strong></td>
<td style="text-align: center;">0.101</td>
</tr>
</tbody>
</table>
</div>
</div>
