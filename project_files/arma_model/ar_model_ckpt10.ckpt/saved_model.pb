 
µ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.9.12v2.9.0-18-gd8ce9f9c3018ËÀ

 Adam/lstm_74/lstm_cell_74/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_74/lstm_cell_74/bias/v

4Adam/lstm_74/lstm_cell_74/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_74/lstm_cell_74/bias/v*
_output_shapes	
:*
dtype0
¶
,Adam/lstm_74/lstm_cell_74/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/lstm_74/lstm_cell_74/recurrent_kernel/v
¯
@Adam/lstm_74/lstm_cell_74/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_74/lstm_cell_74/recurrent_kernel/v* 
_output_shapes
:
*
dtype0
¡
"Adam/lstm_74/lstm_cell_74/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/lstm_74/lstm_cell_74/kernel/v

6Adam/lstm_74/lstm_cell_74/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_74/lstm_cell_74/kernel/v*
_output_shapes
:	*
dtype0

Adam/dense_71/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_71/bias/v
y
(Adam/dense_71/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_71/bias/v*
_output_shapes
:
*
dtype0

Adam/dense_71/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*'
shared_nameAdam/dense_71/kernel/v

*Adam/dense_71/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_71/kernel/v*
_output_shapes
:	
*
dtype0

 Adam/lstm_74/lstm_cell_74/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_74/lstm_cell_74/bias/m

4Adam/lstm_74/lstm_cell_74/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_74/lstm_cell_74/bias/m*
_output_shapes	
:*
dtype0
¶
,Adam/lstm_74/lstm_cell_74/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/lstm_74/lstm_cell_74/recurrent_kernel/m
¯
@Adam/lstm_74/lstm_cell_74/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_74/lstm_cell_74/recurrent_kernel/m* 
_output_shapes
:
*
dtype0
¡
"Adam/lstm_74/lstm_cell_74/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/lstm_74/lstm_cell_74/kernel/m

6Adam/lstm_74/lstm_cell_74/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_74/lstm_cell_74/kernel/m*
_output_shapes
:	*
dtype0

Adam/dense_71/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_71/bias/m
y
(Adam/dense_71/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_71/bias/m*
_output_shapes
:
*
dtype0

Adam/dense_71/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*'
shared_nameAdam/dense_71/kernel/m

*Adam/dense_71/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_71/kernel/m*
_output_shapes
:	
*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

lstm_74/lstm_cell_74/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_74/lstm_cell_74/bias

-lstm_74/lstm_cell_74/bias/Read/ReadVariableOpReadVariableOplstm_74/lstm_cell_74/bias*
_output_shapes	
:*
dtype0
¨
%lstm_74/lstm_cell_74/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%lstm_74/lstm_cell_74/recurrent_kernel
¡
9lstm_74/lstm_cell_74/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_74/lstm_cell_74/recurrent_kernel* 
_output_shapes
:
*
dtype0

lstm_74/lstm_cell_74/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_namelstm_74/lstm_cell_74/kernel

/lstm_74/lstm_cell_74/kernel/Read/ReadVariableOpReadVariableOplstm_74/lstm_cell_74/kernel*
_output_shapes
:	*
dtype0
r
dense_71/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_71/bias
k
!dense_71/bias/Read/ReadVariableOpReadVariableOpdense_71/bias*
_output_shapes
:
*
dtype0
{
dense_71/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
* 
shared_namedense_71/kernel
t
#dense_71/kernel/Read/ReadVariableOpReadVariableOpdense_71/kernel*
_output_shapes
:	
*
dtype0

NoOpNoOp
«4
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*æ3
valueÜ3BÙ3 BÒ3
Á
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator* 
¦
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias*

'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses* 
'
-0
.1
/2
%3
&4*
'
-0
.1
/2
%3
&4*
* 
°
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
5trace_0
6trace_1
7trace_2
8trace_3* 
6
9trace_0
:trace_1
;trace_2
<trace_3* 
* 
¦
=iter

>beta_1

?beta_2
	@decay
Alearning_rate%m~&m-m.m/m%v&v-v.v/v*

Bserving_default* 

-0
.1
/2*

-0
.1
/2*
* 


Cstates
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Itrace_0
Jtrace_1
Ktrace_2
Ltrace_3* 
6
Mtrace_0
Ntrace_1
Otrace_2
Ptrace_3* 
* 
ã
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
W_random_generator
X
state_size

-kernel
.recurrent_kernel
/bias*
* 
* 
* 
* 

Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

^trace_0
_trace_1* 

`trace_0
atrace_1* 
* 

%0
&1*

%0
&1*
* 

bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

gtrace_0* 

htrace_0* 
_Y
VARIABLE_VALUEdense_71/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_71/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses* 

ntrace_0* 

otrace_0* 
[U
VARIABLE_VALUElstm_74/lstm_cell_74/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_74/lstm_cell_74/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_74/lstm_cell_74/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

p0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

-0
.1
/2*

-0
.1
/2*
* 

qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*

vtrace_0
wtrace_1* 

xtrace_0
ytrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
z	variables
{	keras_api
	|total
	}count*
* 
* 
* 
* 
* 
* 
* 
* 
* 

|0
}1*

z	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_71/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_71/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_74/lstm_cell_74/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_74/lstm_cell_74/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_74/lstm_cell_74/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_71/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_71/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_74/lstm_cell_74/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_74/lstm_cell_74/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_74/lstm_cell_74/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_47Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ

Á
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_47lstm_74/lstm_cell_74/kernel%lstm_74/lstm_cell_74/recurrent_kernellstm_74/lstm_cell_74/biasdense_71/kerneldense_71/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_3244321
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 


StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_71/kernel/Read/ReadVariableOp!dense_71/bias/Read/ReadVariableOp/lstm_74/lstm_cell_74/kernel/Read/ReadVariableOp9lstm_74/lstm_cell_74/recurrent_kernel/Read/ReadVariableOp-lstm_74/lstm_cell_74/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_71/kernel/m/Read/ReadVariableOp(Adam/dense_71/bias/m/Read/ReadVariableOp6Adam/lstm_74/lstm_cell_74/kernel/m/Read/ReadVariableOp@Adam/lstm_74/lstm_cell_74/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_74/lstm_cell_74/bias/m/Read/ReadVariableOp*Adam/dense_71/kernel/v/Read/ReadVariableOp(Adam/dense_71/bias/v/Read/ReadVariableOp6Adam/lstm_74/lstm_cell_74/kernel/v/Read/ReadVariableOp@Adam/lstm_74/lstm_cell_74/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_74/lstm_cell_74/bias/v/Read/ReadVariableOpConst*#
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_3245539
Ã
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_71/kerneldense_71/biaslstm_74/lstm_cell_74/kernel%lstm_74/lstm_cell_74/recurrent_kernellstm_74/lstm_cell_74/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_71/kernel/mAdam/dense_71/bias/m"Adam/lstm_74/lstm_cell_74/kernel/m,Adam/lstm_74/lstm_cell_74/recurrent_kernel/m Adam/lstm_74/lstm_cell_74/bias/mAdam/dense_71/kernel/vAdam/dense_71/bias/v"Adam/lstm_74/lstm_cell_74/kernel/v,Adam/lstm_74/lstm_cell_74/recurrent_kernel/v Adam/lstm_74/lstm_cell_74/bias/v*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_3245615ÔÈ
ô

I__inference_lstm_cell_74_layer_call_and_return_conditional_losses_3245418

inputs
states_0
states_11
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
9

D__inference_lstm_74_layer_call_and_return_conditional_losses_3243777

inputs'
lstm_cell_74_3243693:	(
lstm_cell_74_3243695:
#
lstm_cell_74_3243697:	
identity¢$lstm_cell_74/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskü
$lstm_cell_74/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_74_3243693lstm_cell_74_3243695lstm_cell_74_3243697*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_74_layer_call_and_return_conditional_losses_3243647n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : À
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_74_3243693lstm_cell_74_3243695lstm_cell_74_3243697*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3243707*
condR
while_cond_3243706*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
NoOpNoOp%^lstm_cell_74/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_74/StatefulPartitionedCall$lstm_cell_74/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì

I__inference_lstm_cell_74_layer_call_and_return_conditional_losses_3243647

inputs

states
states_11
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
¿9
Ó
while_body_3245062
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_74_matmul_readvariableop_resource_0:	I
5while_lstm_cell_74_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_74_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_74_matmul_readvariableop_resource:	G
3while_lstm_cell_74_matmul_1_readvariableop_resource:
A
2while_lstm_cell_74_biasadd_readvariableop_resource:	¢)while/lstm_cell_74/BiasAdd/ReadVariableOp¢(while/lstm_cell_74/MatMul/ReadVariableOp¢*while/lstm_cell_74/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_74/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_74_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_74/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_74/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_74_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_74/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_74/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/addAddV2#while/lstm_cell_74/MatMul:product:0%while/lstm_cell_74/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_74/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_74_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_74/BiasAddBiasAddwhile/lstm_cell_74/add:z:01while/lstm_cell_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_74/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_74/splitSplit+while/lstm_cell_74/split/split_dim:output:0#while/lstm_cell_74/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_74/SigmoidSigmoid!while/lstm_cell_74/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_74/Sigmoid_1Sigmoid!while/lstm_cell_74/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/mulMul while/lstm_cell_74/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_74/ReluRelu!while/lstm_cell_74/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/mul_1Mulwhile/lstm_cell_74/Sigmoid:y:0%while/lstm_cell_74/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/add_1AddV2while/lstm_cell_74/mul:z:0while/lstm_cell_74/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_74/Sigmoid_2Sigmoid!while/lstm_cell_74/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_74/Relu_1Reluwhile/lstm_cell_74/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/mul_2Mul while/lstm_cell_74/Sigmoid_2:y:0'while/lstm_cell_74/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_74/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_74/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_74/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_74/BiasAdd/ReadVariableOp)^while/lstm_cell_74/MatMul/ReadVariableOp+^while/lstm_cell_74/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_74_biasadd_readvariableop_resource4while_lstm_cell_74_biasadd_readvariableop_resource_0"l
3while_lstm_cell_74_matmul_1_readvariableop_resource5while_lstm_cell_74_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_74_matmul_readvariableop_resource3while_lstm_cell_74_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_74/BiasAdd/ReadVariableOp)while/lstm_cell_74/BiasAdd/ReadVariableOp2T
(while/lstm_cell_74/MatMul/ReadVariableOp(while/lstm_cell_74/MatMul/ReadVariableOp2X
*while/lstm_cell_74/MatMul_1/ReadVariableOp*while/lstm_cell_74/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ü
·
)__inference_lstm_74_layer_call_fn_3244701

inputs
unknown:	
	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_74_layer_call_and_return_conditional_losses_3243937p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
C
Ó

lstm_74_while_body_3244411,
(lstm_74_while_lstm_74_while_loop_counter2
.lstm_74_while_lstm_74_while_maximum_iterations
lstm_74_while_placeholder
lstm_74_while_placeholder_1
lstm_74_while_placeholder_2
lstm_74_while_placeholder_3+
'lstm_74_while_lstm_74_strided_slice_1_0g
clstm_74_while_tensorarrayv2read_tensorlistgetitem_lstm_74_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_74_while_lstm_cell_74_matmul_readvariableop_resource_0:	Q
=lstm_74_while_lstm_cell_74_matmul_1_readvariableop_resource_0:
K
<lstm_74_while_lstm_cell_74_biasadd_readvariableop_resource_0:	
lstm_74_while_identity
lstm_74_while_identity_1
lstm_74_while_identity_2
lstm_74_while_identity_3
lstm_74_while_identity_4
lstm_74_while_identity_5)
%lstm_74_while_lstm_74_strided_slice_1e
alstm_74_while_tensorarrayv2read_tensorlistgetitem_lstm_74_tensorarrayunstack_tensorlistfromtensorL
9lstm_74_while_lstm_cell_74_matmul_readvariableop_resource:	O
;lstm_74_while_lstm_cell_74_matmul_1_readvariableop_resource:
I
:lstm_74_while_lstm_cell_74_biasadd_readvariableop_resource:	¢1lstm_74/while/lstm_cell_74/BiasAdd/ReadVariableOp¢0lstm_74/while/lstm_cell_74/MatMul/ReadVariableOp¢2lstm_74/while/lstm_cell_74/MatMul_1/ReadVariableOp
?lstm_74/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Î
1lstm_74/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_74_while_tensorarrayv2read_tensorlistgetitem_lstm_74_tensorarrayunstack_tensorlistfromtensor_0lstm_74_while_placeholderHlstm_74/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0­
0lstm_74/while/lstm_cell_74/MatMul/ReadVariableOpReadVariableOp;lstm_74_while_lstm_cell_74_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ò
!lstm_74/while/lstm_cell_74/MatMulMatMul8lstm_74/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_74/while/lstm_cell_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
2lstm_74/while/lstm_cell_74/MatMul_1/ReadVariableOpReadVariableOp=lstm_74_while_lstm_cell_74_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¹
#lstm_74/while/lstm_cell_74/MatMul_1MatMullstm_74_while_placeholder_2:lstm_74/while/lstm_cell_74/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm_74/while/lstm_cell_74/addAddV2+lstm_74/while/lstm_cell_74/MatMul:product:0-lstm_74/while/lstm_cell_74/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
1lstm_74/while/lstm_cell_74/BiasAdd/ReadVariableOpReadVariableOp<lstm_74_while_lstm_cell_74_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¿
"lstm_74/while/lstm_cell_74/BiasAddBiasAdd"lstm_74/while/lstm_cell_74/add:z:09lstm_74/while/lstm_cell_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*lstm_74/while/lstm_cell_74/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_74/while/lstm_cell_74/splitSplit3lstm_74/while/lstm_cell_74/split/split_dim:output:0+lstm_74/while/lstm_cell_74/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"lstm_74/while/lstm_cell_74/SigmoidSigmoid)lstm_74/while/lstm_cell_74/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_74/while/lstm_cell_74/Sigmoid_1Sigmoid)lstm_74/while/lstm_cell_74/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_74/while/lstm_cell_74/mulMul(lstm_74/while/lstm_cell_74/Sigmoid_1:y:0lstm_74_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_74/while/lstm_cell_74/ReluRelu)lstm_74/while/lstm_cell_74/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
 lstm_74/while/lstm_cell_74/mul_1Mul&lstm_74/while/lstm_cell_74/Sigmoid:y:0-lstm_74/while/lstm_cell_74/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
 lstm_74/while/lstm_cell_74/add_1AddV2"lstm_74/while/lstm_cell_74/mul:z:0$lstm_74/while/lstm_cell_74/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_74/while/lstm_cell_74/Sigmoid_2Sigmoid)lstm_74/while/lstm_cell_74/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_74/while/lstm_cell_74/Relu_1Relu$lstm_74/while/lstm_cell_74/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
 lstm_74/while/lstm_cell_74/mul_2Mul(lstm_74/while/lstm_cell_74/Sigmoid_2:y:0/lstm_74/while/lstm_cell_74/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8lstm_74/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
2lstm_74/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_74_while_placeholder_1Alstm_74/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_74/while/lstm_cell_74/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_74/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_74/while/addAddV2lstm_74_while_placeholderlstm_74/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_74/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_74/while/add_1AddV2(lstm_74_while_lstm_74_while_loop_counterlstm_74/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_74/while/IdentityIdentitylstm_74/while/add_1:z:0^lstm_74/while/NoOp*
T0*
_output_shapes
: 
lstm_74/while/Identity_1Identity.lstm_74_while_lstm_74_while_maximum_iterations^lstm_74/while/NoOp*
T0*
_output_shapes
: q
lstm_74/while/Identity_2Identitylstm_74/while/add:z:0^lstm_74/while/NoOp*
T0*
_output_shapes
: 
lstm_74/while/Identity_3IdentityBlstm_74/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_74/while/NoOp*
T0*
_output_shapes
: 
lstm_74/while/Identity_4Identity$lstm_74/while/lstm_cell_74/mul_2:z:0^lstm_74/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_74/while/Identity_5Identity$lstm_74/while/lstm_cell_74/add_1:z:0^lstm_74/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
lstm_74/while/NoOpNoOp2^lstm_74/while/lstm_cell_74/BiasAdd/ReadVariableOp1^lstm_74/while/lstm_cell_74/MatMul/ReadVariableOp3^lstm_74/while/lstm_cell_74/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_74_while_identitylstm_74/while/Identity:output:0"=
lstm_74_while_identity_1!lstm_74/while/Identity_1:output:0"=
lstm_74_while_identity_2!lstm_74/while/Identity_2:output:0"=
lstm_74_while_identity_3!lstm_74/while/Identity_3:output:0"=
lstm_74_while_identity_4!lstm_74/while/Identity_4:output:0"=
lstm_74_while_identity_5!lstm_74/while/Identity_5:output:0"P
%lstm_74_while_lstm_74_strided_slice_1'lstm_74_while_lstm_74_strided_slice_1_0"z
:lstm_74_while_lstm_cell_74_biasadd_readvariableop_resource<lstm_74_while_lstm_cell_74_biasadd_readvariableop_resource_0"|
;lstm_74_while_lstm_cell_74_matmul_1_readvariableop_resource=lstm_74_while_lstm_cell_74_matmul_1_readvariableop_resource_0"x
9lstm_74_while_lstm_cell_74_matmul_readvariableop_resource;lstm_74_while_lstm_cell_74_matmul_readvariableop_resource_0"È
alstm_74_while_tensorarrayv2read_tensorlistgetitem_lstm_74_tensorarrayunstack_tensorlistfromtensorclstm_74_while_tensorarrayv2read_tensorlistgetitem_lstm_74_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_74/while/lstm_cell_74/BiasAdd/ReadVariableOp1lstm_74/while/lstm_cell_74/BiasAdd/ReadVariableOp2d
0lstm_74/while/lstm_cell_74/MatMul/ReadVariableOp0lstm_74/while/lstm_cell_74/MatMul/ReadVariableOp2h
2lstm_74/while/lstm_cell_74/MatMul_1/ReadVariableOp2lstm_74/while/lstm_cell_74/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ñj
¹
"__inference__wrapped_model_3243432
input_47M
:ar_mod_lstm_74_lstm_cell_74_matmul_readvariableop_resource:	P
<ar_mod_lstm_74_lstm_cell_74_matmul_1_readvariableop_resource:
J
;ar_mod_lstm_74_lstm_cell_74_biasadd_readvariableop_resource:	A
.ar_mod_dense_71_matmul_readvariableop_resource:	
=
/ar_mod_dense_71_biasadd_readvariableop_resource:

identity¢&ar_mod/dense_71/BiasAdd/ReadVariableOp¢%ar_mod/dense_71/MatMul/ReadVariableOp¢2ar_mod/lstm_74/lstm_cell_74/BiasAdd/ReadVariableOp¢1ar_mod/lstm_74/lstm_cell_74/MatMul/ReadVariableOp¢3ar_mod/lstm_74/lstm_cell_74/MatMul_1/ReadVariableOp¢ar_mod/lstm_74/whileL
ar_mod/lstm_74/ShapeShapeinput_47*
T0*
_output_shapes
:l
"ar_mod/lstm_74/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$ar_mod/lstm_74/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$ar_mod/lstm_74/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ar_mod/lstm_74/strided_sliceStridedSlicear_mod/lstm_74/Shape:output:0+ar_mod/lstm_74/strided_slice/stack:output:0-ar_mod/lstm_74/strided_slice/stack_1:output:0-ar_mod/lstm_74/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
ar_mod/lstm_74/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B : 
ar_mod/lstm_74/zeros/packedPack%ar_mod/lstm_74/strided_slice:output:0&ar_mod/lstm_74/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:_
ar_mod/lstm_74/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
ar_mod/lstm_74/zerosFill$ar_mod/lstm_74/zeros/packed:output:0#ar_mod/lstm_74/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
ar_mod/lstm_74/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¤
ar_mod/lstm_74/zeros_1/packedPack%ar_mod/lstm_74/strided_slice:output:0(ar_mod/lstm_74/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:a
ar_mod/lstm_74/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *     
ar_mod/lstm_74/zeros_1Fill&ar_mod/lstm_74/zeros_1/packed:output:0%ar_mod/lstm_74/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
ar_mod/lstm_74/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
ar_mod/lstm_74/transpose	Transposeinput_47&ar_mod/lstm_74/transpose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿb
ar_mod/lstm_74/Shape_1Shapear_mod/lstm_74/transpose:y:0*
T0*
_output_shapes
:n
$ar_mod/lstm_74/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&ar_mod/lstm_74/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&ar_mod/lstm_74/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
ar_mod/lstm_74/strided_slice_1StridedSlicear_mod/lstm_74/Shape_1:output:0-ar_mod/lstm_74/strided_slice_1/stack:output:0/ar_mod/lstm_74/strided_slice_1/stack_1:output:0/ar_mod/lstm_74/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
*ar_mod/lstm_74/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿá
ar_mod/lstm_74/TensorArrayV2TensorListReserve3ar_mod/lstm_74/TensorArrayV2/element_shape:output:0'ar_mod/lstm_74/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Dar_mod/lstm_74/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
6ar_mod/lstm_74/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorar_mod/lstm_74/transpose:y:0Mar_mod/lstm_74/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$ar_mod/lstm_74/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&ar_mod/lstm_74/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&ar_mod/lstm_74/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:´
ar_mod/lstm_74/strided_slice_2StridedSlicear_mod/lstm_74/transpose:y:0-ar_mod/lstm_74/strided_slice_2/stack:output:0/ar_mod/lstm_74/strided_slice_2/stack_1:output:0/ar_mod/lstm_74/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask­
1ar_mod/lstm_74/lstm_cell_74/MatMul/ReadVariableOpReadVariableOp:ar_mod_lstm_74_lstm_cell_74_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Ã
"ar_mod/lstm_74/lstm_cell_74/MatMulMatMul'ar_mod/lstm_74/strided_slice_2:output:09ar_mod/lstm_74/lstm_cell_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
3ar_mod/lstm_74/lstm_cell_74/MatMul_1/ReadVariableOpReadVariableOp<ar_mod_lstm_74_lstm_cell_74_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0½
$ar_mod/lstm_74/lstm_cell_74/MatMul_1MatMular_mod/lstm_74/zeros:output:0;ar_mod/lstm_74/lstm_cell_74/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
ar_mod/lstm_74/lstm_cell_74/addAddV2,ar_mod/lstm_74/lstm_cell_74/MatMul:product:0.ar_mod/lstm_74/lstm_cell_74/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
2ar_mod/lstm_74/lstm_cell_74/BiasAdd/ReadVariableOpReadVariableOp;ar_mod_lstm_74_lstm_cell_74_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Â
#ar_mod/lstm_74/lstm_cell_74/BiasAddBiasAdd#ar_mod/lstm_74/lstm_cell_74/add:z:0:ar_mod/lstm_74/lstm_cell_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
+ar_mod/lstm_74/lstm_cell_74/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!ar_mod/lstm_74/lstm_cell_74/splitSplit4ar_mod/lstm_74/lstm_cell_74/split/split_dim:output:0,ar_mod/lstm_74/lstm_cell_74/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
#ar_mod/lstm_74/lstm_cell_74/SigmoidSigmoid*ar_mod/lstm_74/lstm_cell_74/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%ar_mod/lstm_74/lstm_cell_74/Sigmoid_1Sigmoid*ar_mod/lstm_74/lstm_cell_74/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
ar_mod/lstm_74/lstm_cell_74/mulMul)ar_mod/lstm_74/lstm_cell_74/Sigmoid_1:y:0ar_mod/lstm_74/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 ar_mod/lstm_74/lstm_cell_74/ReluRelu*ar_mod/lstm_74/lstm_cell_74/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
!ar_mod/lstm_74/lstm_cell_74/mul_1Mul'ar_mod/lstm_74/lstm_cell_74/Sigmoid:y:0.ar_mod/lstm_74/lstm_cell_74/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
!ar_mod/lstm_74/lstm_cell_74/add_1AddV2#ar_mod/lstm_74/lstm_cell_74/mul:z:0%ar_mod/lstm_74/lstm_cell_74/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%ar_mod/lstm_74/lstm_cell_74/Sigmoid_2Sigmoid*ar_mod/lstm_74/lstm_cell_74/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"ar_mod/lstm_74/lstm_cell_74/Relu_1Relu%ar_mod/lstm_74/lstm_cell_74/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
!ar_mod/lstm_74/lstm_cell_74/mul_2Mul)ar_mod/lstm_74/lstm_cell_74/Sigmoid_2:y:00ar_mod/lstm_74/lstm_cell_74/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
,ar_mod/lstm_74/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   m
+ar_mod/lstm_74/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ò
ar_mod/lstm_74/TensorArrayV2_1TensorListReserve5ar_mod/lstm_74/TensorArrayV2_1/element_shape:output:04ar_mod/lstm_74/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒU
ar_mod/lstm_74/timeConst*
_output_shapes
: *
dtype0*
value	B : r
'ar_mod/lstm_74/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿc
!ar_mod/lstm_74/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ú
ar_mod/lstm_74/whileWhile*ar_mod/lstm_74/while/loop_counter:output:00ar_mod/lstm_74/while/maximum_iterations:output:0ar_mod/lstm_74/time:output:0'ar_mod/lstm_74/TensorArrayV2_1:handle:0ar_mod/lstm_74/zeros:output:0ar_mod/lstm_74/zeros_1:output:0'ar_mod/lstm_74/strided_slice_1:output:0Far_mod/lstm_74/TensorArrayUnstack/TensorListFromTensor:output_handle:0:ar_mod_lstm_74_lstm_cell_74_matmul_readvariableop_resource<ar_mod_lstm_74_lstm_cell_74_matmul_1_readvariableop_resource;ar_mod_lstm_74_lstm_cell_74_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *-
body%R#
!ar_mod_lstm_74_while_body_3243337*-
cond%R#
!ar_mod_lstm_74_while_cond_3243336*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
?ar_mod/lstm_74/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
1ar_mod/lstm_74/TensorArrayV2Stack/TensorListStackTensorListStackar_mod/lstm_74/while:output:3Har_mod/lstm_74/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsw
$ar_mod/lstm_74/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿp
&ar_mod/lstm_74/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&ar_mod/lstm_74/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
ar_mod/lstm_74/strided_slice_3StridedSlice:ar_mod/lstm_74/TensorArrayV2Stack/TensorListStack:tensor:0-ar_mod/lstm_74/strided_slice_3/stack:output:0/ar_mod/lstm_74/strided_slice_3/stack_1:output:0/ar_mod/lstm_74/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskt
ar_mod/lstm_74/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ä
ar_mod/lstm_74/transpose_1	Transpose:ar_mod/lstm_74/TensorArrayV2Stack/TensorListStack:tensor:0(ar_mod/lstm_74/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
ar_mod/lstm_74/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
ar_mod/dropout_7/IdentityIdentity'ar_mod/lstm_74/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%ar_mod/dense_71/MatMul/ReadVariableOpReadVariableOp.ar_mod_dense_71_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0¥
ar_mod/dense_71/MatMulMatMul"ar_mod/dropout_7/Identity:output:0-ar_mod/dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&ar_mod/dense_71/BiasAdd/ReadVariableOpReadVariableOp/ar_mod_dense_71_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0¦
ar_mod/dense_71/BiasAddBiasAdd ar_mod/dense_71/MatMul:product:0.ar_mod/dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
&ar_mod/weighted_layer_32/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   °
 ar_mod/weighted_layer_32/ReshapeReshape ar_mod/dense_71/BiasAdd:output:0/ar_mod/weighted_layer_32/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ar_mod/weighted_layer_32/MulMulinput_47)ar_mod/weighted_layer_32/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
IdentityIdentity ar_mod/weighted_layer_32/Mul:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Í
NoOpNoOp'^ar_mod/dense_71/BiasAdd/ReadVariableOp&^ar_mod/dense_71/MatMul/ReadVariableOp3^ar_mod/lstm_74/lstm_cell_74/BiasAdd/ReadVariableOp2^ar_mod/lstm_74/lstm_cell_74/MatMul/ReadVariableOp4^ar_mod/lstm_74/lstm_cell_74/MatMul_1/ReadVariableOp^ar_mod/lstm_74/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ
: : : : : 2P
&ar_mod/dense_71/BiasAdd/ReadVariableOp&ar_mod/dense_71/BiasAdd/ReadVariableOp2N
%ar_mod/dense_71/MatMul/ReadVariableOp%ar_mod/dense_71/MatMul/ReadVariableOp2h
2ar_mod/lstm_74/lstm_cell_74/BiasAdd/ReadVariableOp2ar_mod/lstm_74/lstm_cell_74/BiasAdd/ReadVariableOp2f
1ar_mod/lstm_74/lstm_cell_74/MatMul/ReadVariableOp1ar_mod/lstm_74/lstm_cell_74/MatMul/ReadVariableOp2j
3ar_mod/lstm_74/lstm_cell_74/MatMul_1/ReadVariableOp3ar_mod/lstm_74/lstm_cell_74/MatMul_1/ReadVariableOp2,
ar_mod/lstm_74/whilear_mod/lstm_74/while:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
input_47
[
¯
#__inference__traced_restore_3245615
file_prefix3
 assignvariableop_dense_71_kernel:	
.
 assignvariableop_1_dense_71_bias:
A
.assignvariableop_2_lstm_74_lstm_cell_74_kernel:	L
8assignvariableop_3_lstm_74_lstm_cell_74_recurrent_kernel:
;
,assignvariableop_4_lstm_74_lstm_cell_74_bias:	&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: #
assignvariableop_10_total: #
assignvariableop_11_count: =
*assignvariableop_12_adam_dense_71_kernel_m:	
6
(assignvariableop_13_adam_dense_71_bias_m:
I
6assignvariableop_14_adam_lstm_74_lstm_cell_74_kernel_m:	T
@assignvariableop_15_adam_lstm_74_lstm_cell_74_recurrent_kernel_m:
C
4assignvariableop_16_adam_lstm_74_lstm_cell_74_bias_m:	=
*assignvariableop_17_adam_dense_71_kernel_v:	
6
(assignvariableop_18_adam_dense_71_bias_v:
I
6assignvariableop_19_adam_lstm_74_lstm_cell_74_kernel_v:	T
@assignvariableop_20_adam_lstm_74_lstm_cell_74_recurrent_kernel_v:
C
4assignvariableop_21_adam_lstm_74_lstm_cell_74_bias_v:	
identity_23¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¨
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Î

valueÄ
BÁ
B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_dense_71_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_71_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp.assignvariableop_2_lstm_74_lstm_cell_74_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_3AssignVariableOp8assignvariableop_3_lstm_74_lstm_cell_74_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_74_lstm_cell_74_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp*assignvariableop_12_adam_dense_71_kernel_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp(assignvariableop_13_adam_dense_71_bias_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_14AssignVariableOp6assignvariableop_14_adam_lstm_74_lstm_cell_74_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_15AssignVariableOp@assignvariableop_15_adam_lstm_74_lstm_cell_74_recurrent_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_16AssignVariableOp4assignvariableop_16_adam_lstm_74_lstm_cell_74_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_71_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_71_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_lstm_74_lstm_cell_74_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_20AssignVariableOp@assignvariableop_20_adam_lstm_74_lstm_cell_74_recurrent_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_lstm_74_lstm_cell_74_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ³
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
:  
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ó`

C__inference_ar_mod_layer_call_and_return_conditional_losses_3244506

inputsF
3lstm_74_lstm_cell_74_matmul_readvariableop_resource:	I
5lstm_74_lstm_cell_74_matmul_1_readvariableop_resource:
C
4lstm_74_lstm_cell_74_biasadd_readvariableop_resource:	:
'dense_71_matmul_readvariableop_resource:	
6
(dense_71_biasadd_readvariableop_resource:

identity¢dense_71/BiasAdd/ReadVariableOp¢dense_71/MatMul/ReadVariableOp¢+lstm_74/lstm_cell_74/BiasAdd/ReadVariableOp¢*lstm_74/lstm_cell_74/MatMul/ReadVariableOp¢,lstm_74/lstm_cell_74/MatMul_1/ReadVariableOp¢lstm_74/whileC
lstm_74/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_74/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_74/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_74/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_74/strided_sliceStridedSlicelstm_74/Shape:output:0$lstm_74/strided_slice/stack:output:0&lstm_74/strided_slice/stack_1:output:0&lstm_74/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_74/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_74/zeros/packedPacklstm_74/strided_slice:output:0lstm_74/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_74/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_74/zerosFilllstm_74/zeros/packed:output:0lstm_74/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm_74/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_74/zeros_1/packedPacklstm_74/strided_slice:output:0!lstm_74/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_74/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_74/zeros_1Filllstm_74/zeros_1/packed:output:0lstm_74/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_74/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_74/transpose	Transposeinputslstm_74/transpose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿT
lstm_74/Shape_1Shapelstm_74/transpose:y:0*
T0*
_output_shapes
:g
lstm_74/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_74/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_74/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_74/strided_slice_1StridedSlicelstm_74/Shape_1:output:0&lstm_74/strided_slice_1/stack:output:0(lstm_74/strided_slice_1/stack_1:output:0(lstm_74/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_74/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_74/TensorArrayV2TensorListReserve,lstm_74/TensorArrayV2/element_shape:output:0 lstm_74/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_74/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_74/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_74/transpose:y:0Flstm_74/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_74/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_74/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_74/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_74/strided_slice_2StridedSlicelstm_74/transpose:y:0&lstm_74/strided_slice_2/stack:output:0(lstm_74/strided_slice_2/stack_1:output:0(lstm_74/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
*lstm_74/lstm_cell_74/MatMul/ReadVariableOpReadVariableOp3lstm_74_lstm_cell_74_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0®
lstm_74/lstm_cell_74/MatMulMatMul lstm_74/strided_slice_2:output:02lstm_74/lstm_cell_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
,lstm_74/lstm_cell_74/MatMul_1/ReadVariableOpReadVariableOp5lstm_74_lstm_cell_74_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0¨
lstm_74/lstm_cell_74/MatMul_1MatMullstm_74/zeros:output:04lstm_74/lstm_cell_74/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
lstm_74/lstm_cell_74/addAddV2%lstm_74/lstm_cell_74/MatMul:product:0'lstm_74/lstm_cell_74/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+lstm_74/lstm_cell_74/BiasAdd/ReadVariableOpReadVariableOp4lstm_74_lstm_cell_74_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
lstm_74/lstm_cell_74/BiasAddBiasAddlstm_74/lstm_cell_74/add:z:03lstm_74/lstm_cell_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$lstm_74/lstm_cell_74/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ù
lstm_74/lstm_cell_74/splitSplit-lstm_74/lstm_cell_74/split/split_dim:output:0%lstm_74/lstm_cell_74/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
lstm_74/lstm_cell_74/SigmoidSigmoid#lstm_74/lstm_cell_74/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_74/lstm_cell_74/Sigmoid_1Sigmoid#lstm_74/lstm_cell_74/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_74/lstm_cell_74/mulMul"lstm_74/lstm_cell_74/Sigmoid_1:y:0lstm_74/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_74/lstm_cell_74/ReluRelu#lstm_74/lstm_cell_74/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_74/lstm_cell_74/mul_1Mul lstm_74/lstm_cell_74/Sigmoid:y:0'lstm_74/lstm_cell_74/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_74/lstm_cell_74/add_1AddV2lstm_74/lstm_cell_74/mul:z:0lstm_74/lstm_cell_74/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_74/lstm_cell_74/Sigmoid_2Sigmoid#lstm_74/lstm_cell_74/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
lstm_74/lstm_cell_74/Relu_1Relulstm_74/lstm_cell_74/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
lstm_74/lstm_cell_74/mul_2Mul"lstm_74/lstm_cell_74/Sigmoid_2:y:0)lstm_74/lstm_cell_74/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%lstm_74/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   f
$lstm_74/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_74/TensorArrayV2_1TensorListReserve.lstm_74/TensorArrayV2_1/element_shape:output:0-lstm_74/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_74/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_74/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_74/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ø
lstm_74/whileWhile#lstm_74/while/loop_counter:output:0)lstm_74/while/maximum_iterations:output:0lstm_74/time:output:0 lstm_74/TensorArrayV2_1:handle:0lstm_74/zeros:output:0lstm_74/zeros_1:output:0 lstm_74/strided_slice_1:output:0?lstm_74/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_74_lstm_cell_74_matmul_readvariableop_resource5lstm_74_lstm_cell_74_matmul_1_readvariableop_resource4lstm_74_lstm_cell_74_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_74_while_body_3244411*&
condR
lstm_74_while_cond_3244410*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
8lstm_74/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ï
*lstm_74/TensorArrayV2Stack/TensorListStackTensorListStacklstm_74/while:output:3Alstm_74/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsp
lstm_74/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_74/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_74/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
lstm_74/strided_slice_3StridedSlice3lstm_74/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_74/strided_slice_3/stack:output:0(lstm_74/strided_slice_3/stack_1:output:0(lstm_74/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskm
lstm_74/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¯
lstm_74/transpose_1	Transpose3lstm_74/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_74/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_74/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    s
dropout_7/IdentityIdentity lstm_74/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_71/MatMul/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
dense_71/MatMulMatMuldropout_7/Identity:output:0&dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_71/BiasAdd/ReadVariableOpReadVariableOp(dense_71_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_71/BiasAddBiasAdddense_71/MatMul:product:0'dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
weighted_layer_32/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
weighted_layer_32/ReshapeReshapedense_71/BiasAdd:output:0(weighted_layer_32/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
weighted_layer_32/MulMulinputs"weighted_layer_32/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
l
IdentityIdentityweighted_layer_32/Mul:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
£
NoOpNoOp ^dense_71/BiasAdd/ReadVariableOp^dense_71/MatMul/ReadVariableOp,^lstm_74/lstm_cell_74/BiasAdd/ReadVariableOp+^lstm_74/lstm_cell_74/MatMul/ReadVariableOp-^lstm_74/lstm_cell_74/MatMul_1/ReadVariableOp^lstm_74/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ
: : : : : 2B
dense_71/BiasAdd/ReadVariableOpdense_71/BiasAdd/ReadVariableOp2@
dense_71/MatMul/ReadVariableOpdense_71/MatMul/ReadVariableOp2Z
+lstm_74/lstm_cell_74/BiasAdd/ReadVariableOp+lstm_74/lstm_cell_74/BiasAdd/ReadVariableOp2X
*lstm_74/lstm_cell_74/MatMul/ReadVariableOp*lstm_74/lstm_cell_74/MatMul/ReadVariableOp2\
,lstm_74/lstm_cell_74/MatMul_1/ReadVariableOp,lstm_74/lstm_cell_74/MatMul_1/ReadVariableOp2
lstm_74/whilelstm_74/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ì	
÷
E__inference_dense_71_layer_call_and_return_conditional_losses_3245338

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
C
Ó

lstm_74_while_body_3244566,
(lstm_74_while_lstm_74_while_loop_counter2
.lstm_74_while_lstm_74_while_maximum_iterations
lstm_74_while_placeholder
lstm_74_while_placeholder_1
lstm_74_while_placeholder_2
lstm_74_while_placeholder_3+
'lstm_74_while_lstm_74_strided_slice_1_0g
clstm_74_while_tensorarrayv2read_tensorlistgetitem_lstm_74_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_74_while_lstm_cell_74_matmul_readvariableop_resource_0:	Q
=lstm_74_while_lstm_cell_74_matmul_1_readvariableop_resource_0:
K
<lstm_74_while_lstm_cell_74_biasadd_readvariableop_resource_0:	
lstm_74_while_identity
lstm_74_while_identity_1
lstm_74_while_identity_2
lstm_74_while_identity_3
lstm_74_while_identity_4
lstm_74_while_identity_5)
%lstm_74_while_lstm_74_strided_slice_1e
alstm_74_while_tensorarrayv2read_tensorlistgetitem_lstm_74_tensorarrayunstack_tensorlistfromtensorL
9lstm_74_while_lstm_cell_74_matmul_readvariableop_resource:	O
;lstm_74_while_lstm_cell_74_matmul_1_readvariableop_resource:
I
:lstm_74_while_lstm_cell_74_biasadd_readvariableop_resource:	¢1lstm_74/while/lstm_cell_74/BiasAdd/ReadVariableOp¢0lstm_74/while/lstm_cell_74/MatMul/ReadVariableOp¢2lstm_74/while/lstm_cell_74/MatMul_1/ReadVariableOp
?lstm_74/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Î
1lstm_74/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_74_while_tensorarrayv2read_tensorlistgetitem_lstm_74_tensorarrayunstack_tensorlistfromtensor_0lstm_74_while_placeholderHlstm_74/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0­
0lstm_74/while/lstm_cell_74/MatMul/ReadVariableOpReadVariableOp;lstm_74_while_lstm_cell_74_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ò
!lstm_74/while/lstm_cell_74/MatMulMatMul8lstm_74/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_74/while/lstm_cell_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
2lstm_74/while/lstm_cell_74/MatMul_1/ReadVariableOpReadVariableOp=lstm_74_while_lstm_cell_74_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¹
#lstm_74/while/lstm_cell_74/MatMul_1MatMullstm_74_while_placeholder_2:lstm_74/while/lstm_cell_74/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm_74/while/lstm_cell_74/addAddV2+lstm_74/while/lstm_cell_74/MatMul:product:0-lstm_74/while/lstm_cell_74/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
1lstm_74/while/lstm_cell_74/BiasAdd/ReadVariableOpReadVariableOp<lstm_74_while_lstm_cell_74_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¿
"lstm_74/while/lstm_cell_74/BiasAddBiasAdd"lstm_74/while/lstm_cell_74/add:z:09lstm_74/while/lstm_cell_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*lstm_74/while/lstm_cell_74/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_74/while/lstm_cell_74/splitSplit3lstm_74/while/lstm_cell_74/split/split_dim:output:0+lstm_74/while/lstm_cell_74/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"lstm_74/while/lstm_cell_74/SigmoidSigmoid)lstm_74/while/lstm_cell_74/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_74/while/lstm_cell_74/Sigmoid_1Sigmoid)lstm_74/while/lstm_cell_74/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_74/while/lstm_cell_74/mulMul(lstm_74/while/lstm_cell_74/Sigmoid_1:y:0lstm_74_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_74/while/lstm_cell_74/ReluRelu)lstm_74/while/lstm_cell_74/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
 lstm_74/while/lstm_cell_74/mul_1Mul&lstm_74/while/lstm_cell_74/Sigmoid:y:0-lstm_74/while/lstm_cell_74/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
 lstm_74/while/lstm_cell_74/add_1AddV2"lstm_74/while/lstm_cell_74/mul:z:0$lstm_74/while/lstm_cell_74/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_74/while/lstm_cell_74/Sigmoid_2Sigmoid)lstm_74/while/lstm_cell_74/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_74/while/lstm_cell_74/Relu_1Relu$lstm_74/while/lstm_cell_74/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
 lstm_74/while/lstm_cell_74/mul_2Mul(lstm_74/while/lstm_cell_74/Sigmoid_2:y:0/lstm_74/while/lstm_cell_74/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8lstm_74/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
2lstm_74/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_74_while_placeholder_1Alstm_74/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_74/while/lstm_cell_74/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_74/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_74/while/addAddV2lstm_74_while_placeholderlstm_74/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_74/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_74/while/add_1AddV2(lstm_74_while_lstm_74_while_loop_counterlstm_74/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_74/while/IdentityIdentitylstm_74/while/add_1:z:0^lstm_74/while/NoOp*
T0*
_output_shapes
: 
lstm_74/while/Identity_1Identity.lstm_74_while_lstm_74_while_maximum_iterations^lstm_74/while/NoOp*
T0*
_output_shapes
: q
lstm_74/while/Identity_2Identitylstm_74/while/add:z:0^lstm_74/while/NoOp*
T0*
_output_shapes
: 
lstm_74/while/Identity_3IdentityBlstm_74/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_74/while/NoOp*
T0*
_output_shapes
: 
lstm_74/while/Identity_4Identity$lstm_74/while/lstm_cell_74/mul_2:z:0^lstm_74/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_74/while/Identity_5Identity$lstm_74/while/lstm_cell_74/add_1:z:0^lstm_74/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
lstm_74/while/NoOpNoOp2^lstm_74/while/lstm_cell_74/BiasAdd/ReadVariableOp1^lstm_74/while/lstm_cell_74/MatMul/ReadVariableOp3^lstm_74/while/lstm_cell_74/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_74_while_identitylstm_74/while/Identity:output:0"=
lstm_74_while_identity_1!lstm_74/while/Identity_1:output:0"=
lstm_74_while_identity_2!lstm_74/while/Identity_2:output:0"=
lstm_74_while_identity_3!lstm_74/while/Identity_3:output:0"=
lstm_74_while_identity_4!lstm_74/while/Identity_4:output:0"=
lstm_74_while_identity_5!lstm_74/while/Identity_5:output:0"P
%lstm_74_while_lstm_74_strided_slice_1'lstm_74_while_lstm_74_strided_slice_1_0"z
:lstm_74_while_lstm_cell_74_biasadd_readvariableop_resource<lstm_74_while_lstm_cell_74_biasadd_readvariableop_resource_0"|
;lstm_74_while_lstm_cell_74_matmul_1_readvariableop_resource=lstm_74_while_lstm_cell_74_matmul_1_readvariableop_resource_0"x
9lstm_74_while_lstm_cell_74_matmul_readvariableop_resource;lstm_74_while_lstm_cell_74_matmul_readvariableop_resource_0"È
alstm_74_while_tensorarrayv2read_tensorlistgetitem_lstm_74_tensorarrayunstack_tensorlistfromtensorclstm_74_while_tensorarrayv2read_tensorlistgetitem_lstm_74_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_74/while/lstm_cell_74/BiasAdd/ReadVariableOp1lstm_74/while/lstm_cell_74/BiasAdd/ReadVariableOp2d
0lstm_74/while/lstm_cell_74/MatMul/ReadVariableOp0lstm_74/while/lstm_cell_74/MatMul/ReadVariableOp2h
2lstm_74/while/lstm_cell_74/MatMul_1/ReadVariableOp2lstm_74/while/lstm_cell_74/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

¹
)__inference_lstm_74_layer_call_fn_3244679
inputs_0
unknown:	
	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_74_layer_call_and_return_conditional_losses_3243584p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
¿9
Ó
while_body_3245207
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_74_matmul_readvariableop_resource_0:	I
5while_lstm_cell_74_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_74_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_74_matmul_readvariableop_resource:	G
3while_lstm_cell_74_matmul_1_readvariableop_resource:
A
2while_lstm_cell_74_biasadd_readvariableop_resource:	¢)while/lstm_cell_74/BiasAdd/ReadVariableOp¢(while/lstm_cell_74/MatMul/ReadVariableOp¢*while/lstm_cell_74/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_74/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_74_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_74/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_74/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_74_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_74/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_74/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/addAddV2#while/lstm_cell_74/MatMul:product:0%while/lstm_cell_74/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_74/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_74_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_74/BiasAddBiasAddwhile/lstm_cell_74/add:z:01while/lstm_cell_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_74/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_74/splitSplit+while/lstm_cell_74/split/split_dim:output:0#while/lstm_cell_74/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_74/SigmoidSigmoid!while/lstm_cell_74/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_74/Sigmoid_1Sigmoid!while/lstm_cell_74/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/mulMul while/lstm_cell_74/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_74/ReluRelu!while/lstm_cell_74/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/mul_1Mulwhile/lstm_cell_74/Sigmoid:y:0%while/lstm_cell_74/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/add_1AddV2while/lstm_cell_74/mul:z:0while/lstm_cell_74/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_74/Sigmoid_2Sigmoid!while/lstm_cell_74/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_74/Relu_1Reluwhile/lstm_cell_74/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/mul_2Mul while/lstm_cell_74/Sigmoid_2:y:0'while/lstm_cell_74/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_74/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_74/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_74/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_74/BiasAdd/ReadVariableOp)^while/lstm_cell_74/MatMul/ReadVariableOp+^while/lstm_cell_74/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_74_biasadd_readvariableop_resource4while_lstm_cell_74_biasadd_readvariableop_resource_0"l
3while_lstm_cell_74_matmul_1_readvariableop_resource5while_lstm_cell_74_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_74_matmul_readvariableop_resource3while_lstm_cell_74_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_74/BiasAdd/ReadVariableOp)while/lstm_cell_74/BiasAdd/ReadVariableOp2T
(while/lstm_cell_74/MatMul/ReadVariableOp(while/lstm_cell_74/MatMul/ReadVariableOp2X
*while/lstm_cell_74/MatMul_1/ReadVariableOp*while/lstm_cell_74/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 


è
lstm_74_while_cond_3244565,
(lstm_74_while_lstm_74_while_loop_counter2
.lstm_74_while_lstm_74_while_maximum_iterations
lstm_74_while_placeholder
lstm_74_while_placeholder_1
lstm_74_while_placeholder_2
lstm_74_while_placeholder_3.
*lstm_74_while_less_lstm_74_strided_slice_1E
Alstm_74_while_lstm_74_while_cond_3244565___redundant_placeholder0E
Alstm_74_while_lstm_74_while_cond_3244565___redundant_placeholder1E
Alstm_74_while_lstm_74_while_cond_3244565___redundant_placeholder2E
Alstm_74_while_lstm_74_while_cond_3244565___redundant_placeholder3
lstm_74_while_identity

lstm_74/while/LessLesslstm_74_while_placeholder*lstm_74_while_less_lstm_74_strided_slice_1*
T0*
_output_shapes
: [
lstm_74/while/IdentityIdentitylstm_74/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_74_while_identitylstm_74/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Û
ï
(__inference_ar_mod_layer_call_fn_3244262
input_47
unknown:	
	unknown_0:

	unknown_1:	
	unknown_2:	

	unknown_3:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_47unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_ar_mod_layer_call_and_return_conditional_losses_3244234s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ
: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
input_47
Ý
d
F__inference_dropout_7_layer_call_and_return_conditional_losses_3243950

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô

I__inference_lstm_cell_74_layer_call_and_return_conditional_losses_3245450

inputs
states_0
states_11
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ã
ô
!ar_mod_lstm_74_while_cond_3243336:
6ar_mod_lstm_74_while_ar_mod_lstm_74_while_loop_counter@
<ar_mod_lstm_74_while_ar_mod_lstm_74_while_maximum_iterations$
 ar_mod_lstm_74_while_placeholder&
"ar_mod_lstm_74_while_placeholder_1&
"ar_mod_lstm_74_while_placeholder_2&
"ar_mod_lstm_74_while_placeholder_3<
8ar_mod_lstm_74_while_less_ar_mod_lstm_74_strided_slice_1S
Oar_mod_lstm_74_while_ar_mod_lstm_74_while_cond_3243336___redundant_placeholder0S
Oar_mod_lstm_74_while_ar_mod_lstm_74_while_cond_3243336___redundant_placeholder1S
Oar_mod_lstm_74_while_ar_mod_lstm_74_while_cond_3243336___redundant_placeholder2S
Oar_mod_lstm_74_while_ar_mod_lstm_74_while_cond_3243336___redundant_placeholder3!
ar_mod_lstm_74_while_identity

ar_mod/lstm_74/while/LessLess ar_mod_lstm_74_while_placeholder8ar_mod_lstm_74_while_less_ar_mod_lstm_74_strided_slice_1*
T0*
_output_shapes
: i
ar_mod/lstm_74/while/IdentityIdentityar_mod/lstm_74/while/Less:z:0*
T0
*
_output_shapes
: "G
ar_mod_lstm_74_while_identity&ar_mod/lstm_74/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
$
ì
while_body_3243707
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_74_3243731_0:	0
while_lstm_cell_74_3243733_0:
+
while_lstm_cell_74_3243735_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_74_3243731:	.
while_lstm_cell_74_3243733:
)
while_lstm_cell_74_3243735:	¢*while/lstm_cell_74/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0º
*while/lstm_cell_74/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_74_3243731_0while_lstm_cell_74_3243733_0while_lstm_cell_74_3243735_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_74_layer_call_and_return_conditional_losses_3243647r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_74/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity3while/lstm_cell_74/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity3while/lstm_cell_74/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

while/NoOpNoOp+^while/lstm_cell_74/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_74_3243731while_lstm_cell_74_3243731_0":
while_lstm_cell_74_3243733while_lstm_cell_74_3243733_0":
while_lstm_cell_74_3243735while_lstm_cell_74_3243735_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_74/StatefulPartitionedCall*while/lstm_cell_74/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ç

*__inference_dense_71_layer_call_fn_3245328

inputs
unknown:	

	unknown_0:

identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_71_layer_call_and_return_conditional_losses_3243962o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
d
F__inference_dropout_7_layer_call_and_return_conditional_losses_3245307

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Íh

C__inference_ar_mod_layer_call_and_return_conditional_losses_3244668

inputsF
3lstm_74_lstm_cell_74_matmul_readvariableop_resource:	I
5lstm_74_lstm_cell_74_matmul_1_readvariableop_resource:
C
4lstm_74_lstm_cell_74_biasadd_readvariableop_resource:	:
'dense_71_matmul_readvariableop_resource:	
6
(dense_71_biasadd_readvariableop_resource:

identity¢dense_71/BiasAdd/ReadVariableOp¢dense_71/MatMul/ReadVariableOp¢+lstm_74/lstm_cell_74/BiasAdd/ReadVariableOp¢*lstm_74/lstm_cell_74/MatMul/ReadVariableOp¢,lstm_74/lstm_cell_74/MatMul_1/ReadVariableOp¢lstm_74/whileC
lstm_74/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_74/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_74/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_74/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_74/strided_sliceStridedSlicelstm_74/Shape:output:0$lstm_74/strided_slice/stack:output:0&lstm_74/strided_slice/stack_1:output:0&lstm_74/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_74/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_74/zeros/packedPacklstm_74/strided_slice:output:0lstm_74/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_74/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_74/zerosFilllstm_74/zeros/packed:output:0lstm_74/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm_74/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_74/zeros_1/packedPacklstm_74/strided_slice:output:0!lstm_74/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_74/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_74/zeros_1Filllstm_74/zeros_1/packed:output:0lstm_74/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_74/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_74/transpose	Transposeinputslstm_74/transpose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿT
lstm_74/Shape_1Shapelstm_74/transpose:y:0*
T0*
_output_shapes
:g
lstm_74/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_74/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_74/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_74/strided_slice_1StridedSlicelstm_74/Shape_1:output:0&lstm_74/strided_slice_1/stack:output:0(lstm_74/strided_slice_1/stack_1:output:0(lstm_74/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_74/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_74/TensorArrayV2TensorListReserve,lstm_74/TensorArrayV2/element_shape:output:0 lstm_74/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_74/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_74/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_74/transpose:y:0Flstm_74/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_74/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_74/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_74/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_74/strided_slice_2StridedSlicelstm_74/transpose:y:0&lstm_74/strided_slice_2/stack:output:0(lstm_74/strided_slice_2/stack_1:output:0(lstm_74/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
*lstm_74/lstm_cell_74/MatMul/ReadVariableOpReadVariableOp3lstm_74_lstm_cell_74_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0®
lstm_74/lstm_cell_74/MatMulMatMul lstm_74/strided_slice_2:output:02lstm_74/lstm_cell_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
,lstm_74/lstm_cell_74/MatMul_1/ReadVariableOpReadVariableOp5lstm_74_lstm_cell_74_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0¨
lstm_74/lstm_cell_74/MatMul_1MatMullstm_74/zeros:output:04lstm_74/lstm_cell_74/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
lstm_74/lstm_cell_74/addAddV2%lstm_74/lstm_cell_74/MatMul:product:0'lstm_74/lstm_cell_74/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+lstm_74/lstm_cell_74/BiasAdd/ReadVariableOpReadVariableOp4lstm_74_lstm_cell_74_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
lstm_74/lstm_cell_74/BiasAddBiasAddlstm_74/lstm_cell_74/add:z:03lstm_74/lstm_cell_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$lstm_74/lstm_cell_74/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ù
lstm_74/lstm_cell_74/splitSplit-lstm_74/lstm_cell_74/split/split_dim:output:0%lstm_74/lstm_cell_74/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
lstm_74/lstm_cell_74/SigmoidSigmoid#lstm_74/lstm_cell_74/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_74/lstm_cell_74/Sigmoid_1Sigmoid#lstm_74/lstm_cell_74/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_74/lstm_cell_74/mulMul"lstm_74/lstm_cell_74/Sigmoid_1:y:0lstm_74/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_74/lstm_cell_74/ReluRelu#lstm_74/lstm_cell_74/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_74/lstm_cell_74/mul_1Mul lstm_74/lstm_cell_74/Sigmoid:y:0'lstm_74/lstm_cell_74/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_74/lstm_cell_74/add_1AddV2lstm_74/lstm_cell_74/mul:z:0lstm_74/lstm_cell_74/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_74/lstm_cell_74/Sigmoid_2Sigmoid#lstm_74/lstm_cell_74/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
lstm_74/lstm_cell_74/Relu_1Relulstm_74/lstm_cell_74/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
lstm_74/lstm_cell_74/mul_2Mul"lstm_74/lstm_cell_74/Sigmoid_2:y:0)lstm_74/lstm_cell_74/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%lstm_74/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   f
$lstm_74/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_74/TensorArrayV2_1TensorListReserve.lstm_74/TensorArrayV2_1/element_shape:output:0-lstm_74/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_74/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_74/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_74/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ø
lstm_74/whileWhile#lstm_74/while/loop_counter:output:0)lstm_74/while/maximum_iterations:output:0lstm_74/time:output:0 lstm_74/TensorArrayV2_1:handle:0lstm_74/zeros:output:0lstm_74/zeros_1:output:0 lstm_74/strided_slice_1:output:0?lstm_74/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_74_lstm_cell_74_matmul_readvariableop_resource5lstm_74_lstm_cell_74_matmul_1_readvariableop_resource4lstm_74_lstm_cell_74_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_74_while_body_3244566*&
condR
lstm_74_while_cond_3244565*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
8lstm_74/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ï
*lstm_74/TensorArrayV2Stack/TensorListStackTensorListStacklstm_74/while:output:3Alstm_74/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsp
lstm_74/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_74/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_74/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
lstm_74/strided_slice_3StridedSlice3lstm_74/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_74/strided_slice_3/stack:output:0(lstm_74/strided_slice_3/stack_1:output:0(lstm_74/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskm
lstm_74/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¯
lstm_74/transpose_1	Transpose3lstm_74/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_74/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_74/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    \
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_7/dropout/MulMul lstm_74/strided_slice_3:output:0 dropout_7/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
dropout_7/dropout/ShapeShape lstm_74/strided_slice_3:output:0*
T0*
_output_shapes
:®
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seedÒ	e
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Å
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_71/MatMul/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
dense_71/MatMulMatMuldropout_7/dropout/Mul_1:z:0&dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_71/BiasAdd/ReadVariableOpReadVariableOp(dense_71_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_71/BiasAddBiasAdddense_71/MatMul:product:0'dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
weighted_layer_32/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
weighted_layer_32/ReshapeReshapedense_71/BiasAdd:output:0(weighted_layer_32/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
weighted_layer_32/MulMulinputs"weighted_layer_32/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
l
IdentityIdentityweighted_layer_32/Mul:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
£
NoOpNoOp ^dense_71/BiasAdd/ReadVariableOp^dense_71/MatMul/ReadVariableOp,^lstm_74/lstm_cell_74/BiasAdd/ReadVariableOp+^lstm_74/lstm_cell_74/MatMul/ReadVariableOp-^lstm_74/lstm_cell_74/MatMul_1/ReadVariableOp^lstm_74/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ
: : : : : 2B
dense_71/BiasAdd/ReadVariableOpdense_71/BiasAdd/ReadVariableOp2@
dense_71/MatMul/ReadVariableOpdense_71/MatMul/ReadVariableOp2Z
+lstm_74/lstm_cell_74/BiasAdd/ReadVariableOp+lstm_74/lstm_cell_74/BiasAdd/ReadVariableOp2X
*lstm_74/lstm_cell_74/MatMul/ReadVariableOp*lstm_74/lstm_cell_74/MatMul/ReadVariableOp2\
,lstm_74/lstm_cell_74/MatMul_1/ReadVariableOp,lstm_74/lstm_cell_74/MatMul_1/ReadVariableOp2
lstm_74/whilelstm_74/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¾
È
while_cond_3243851
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3243851___redundant_placeholder05
1while_while_cond_3243851___redundant_placeholder15
1while_while_cond_3243851___redundant_placeholder25
1while_while_cond_3243851___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¿9
Ó
while_body_3244105
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_74_matmul_readvariableop_resource_0:	I
5while_lstm_cell_74_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_74_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_74_matmul_readvariableop_resource:	G
3while_lstm_cell_74_matmul_1_readvariableop_resource:
A
2while_lstm_cell_74_biasadd_readvariableop_resource:	¢)while/lstm_cell_74/BiasAdd/ReadVariableOp¢(while/lstm_cell_74/MatMul/ReadVariableOp¢*while/lstm_cell_74/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_74/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_74_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_74/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_74/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_74_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_74/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_74/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/addAddV2#while/lstm_cell_74/MatMul:product:0%while/lstm_cell_74/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_74/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_74_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_74/BiasAddBiasAddwhile/lstm_cell_74/add:z:01while/lstm_cell_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_74/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_74/splitSplit+while/lstm_cell_74/split/split_dim:output:0#while/lstm_cell_74/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_74/SigmoidSigmoid!while/lstm_cell_74/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_74/Sigmoid_1Sigmoid!while/lstm_cell_74/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/mulMul while/lstm_cell_74/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_74/ReluRelu!while/lstm_cell_74/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/mul_1Mulwhile/lstm_cell_74/Sigmoid:y:0%while/lstm_cell_74/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/add_1AddV2while/lstm_cell_74/mul:z:0while/lstm_cell_74/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_74/Sigmoid_2Sigmoid!while/lstm_cell_74/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_74/Relu_1Reluwhile/lstm_cell_74/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/mul_2Mul while/lstm_cell_74/Sigmoid_2:y:0'while/lstm_cell_74/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_74/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_74/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_74/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_74/BiasAdd/ReadVariableOp)^while/lstm_cell_74/MatMul/ReadVariableOp+^while/lstm_cell_74/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_74_biasadd_readvariableop_resource4while_lstm_cell_74_biasadd_readvariableop_resource_0"l
3while_lstm_cell_74_matmul_1_readvariableop_resource5while_lstm_cell_74_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_74_matmul_readvariableop_resource3while_lstm_cell_74_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_74/BiasAdd/ReadVariableOp)while/lstm_cell_74/BiasAdd/ReadVariableOp2T
(while/lstm_cell_74/MatMul/ReadVariableOp(while/lstm_cell_74/MatMul/ReadVariableOp2X
*while/lstm_cell_74/MatMul_1/ReadVariableOp*while/lstm_cell_74/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ÇK

D__inference_lstm_74_layer_call_and_return_conditional_losses_3243937

inputs>
+lstm_cell_74_matmul_readvariableop_resource:	A
-lstm_cell_74_matmul_1_readvariableop_resource:
;
,lstm_cell_74_biasadd_readvariableop_resource:	
identity¢#lstm_cell_74/BiasAdd/ReadVariableOp¢"lstm_cell_74/MatMul/ReadVariableOp¢$lstm_cell_74/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_74/MatMul/ReadVariableOpReadVariableOp+lstm_cell_74_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_74/MatMulMatMulstrided_slice_2:output:0*lstm_cell_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_74/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_74_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_74/MatMul_1MatMulzeros:output:0,lstm_cell_74/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_74/addAddV2lstm_cell_74/MatMul:product:0lstm_cell_74/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_74/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_74_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_74/BiasAddBiasAddlstm_cell_74/add:z:0+lstm_cell_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_74/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_74/splitSplit%lstm_cell_74/split/split_dim:output:0lstm_cell_74/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_74/SigmoidSigmoidlstm_cell_74/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_74/Sigmoid_1Sigmoidlstm_cell_74/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_74/mulMullstm_cell_74/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_74/ReluRelulstm_cell_74/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_74/mul_1Mullstm_cell_74/Sigmoid:y:0lstm_cell_74/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_74/add_1AddV2lstm_cell_74/mul:z:0lstm_cell_74/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_74/Sigmoid_2Sigmoidlstm_cell_74/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_74/Relu_1Relulstm_cell_74/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_74/mul_2Mullstm_cell_74/Sigmoid_2:y:0!lstm_cell_74/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_74_matmul_readvariableop_resource-lstm_cell_74_matmul_1_readvariableop_resource,lstm_cell_74_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3243852*
condR
while_cond_3243851*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_74/BiasAdd/ReadVariableOp#^lstm_cell_74/MatMul/ReadVariableOp%^lstm_cell_74/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 2J
#lstm_cell_74/BiasAdd/ReadVariableOp#lstm_cell_74/BiasAdd/ReadVariableOp2H
"lstm_cell_74/MatMul/ReadVariableOp"lstm_cell_74/MatMul/ReadVariableOp2L
$lstm_cell_74/MatMul_1/ReadVariableOp$lstm_cell_74/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
·
ì
%__inference_signature_wrapper_3244321
input_47
unknown:	
	unknown_0:

	unknown_1:	
	unknown_2:	

	unknown_3:

identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinput_47unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_3243432s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ
: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
input_47
¿9
Ó
while_body_3243852
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_74_matmul_readvariableop_resource_0:	I
5while_lstm_cell_74_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_74_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_74_matmul_readvariableop_resource:	G
3while_lstm_cell_74_matmul_1_readvariableop_resource:
A
2while_lstm_cell_74_biasadd_readvariableop_resource:	¢)while/lstm_cell_74/BiasAdd/ReadVariableOp¢(while/lstm_cell_74/MatMul/ReadVariableOp¢*while/lstm_cell_74/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_74/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_74_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_74/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_74/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_74_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_74/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_74/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/addAddV2#while/lstm_cell_74/MatMul:product:0%while/lstm_cell_74/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_74/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_74_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_74/BiasAddBiasAddwhile/lstm_cell_74/add:z:01while/lstm_cell_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_74/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_74/splitSplit+while/lstm_cell_74/split/split_dim:output:0#while/lstm_cell_74/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_74/SigmoidSigmoid!while/lstm_cell_74/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_74/Sigmoid_1Sigmoid!while/lstm_cell_74/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/mulMul while/lstm_cell_74/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_74/ReluRelu!while/lstm_cell_74/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/mul_1Mulwhile/lstm_cell_74/Sigmoid:y:0%while/lstm_cell_74/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/add_1AddV2while/lstm_cell_74/mul:z:0while/lstm_cell_74/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_74/Sigmoid_2Sigmoid!while/lstm_cell_74/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_74/Relu_1Reluwhile/lstm_cell_74/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/mul_2Mul while/lstm_cell_74/Sigmoid_2:y:0'while/lstm_cell_74/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_74/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_74/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_74/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_74/BiasAdd/ReadVariableOp)^while/lstm_cell_74/MatMul/ReadVariableOp+^while/lstm_cell_74/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_74_biasadd_readvariableop_resource4while_lstm_cell_74_biasadd_readvariableop_resource_0"l
3while_lstm_cell_74_matmul_1_readvariableop_resource5while_lstm_cell_74_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_74_matmul_readvariableop_resource3while_lstm_cell_74_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_74/BiasAdd/ReadVariableOp)while/lstm_cell_74/BiasAdd/ReadVariableOp2T
(while/lstm_cell_74/MatMul/ReadVariableOp(while/lstm_cell_74/MatMul/ReadVariableOp2X
*while/lstm_cell_74/MatMul_1/ReadVariableOp*while/lstm_cell_74/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Õ
í
(__inference_ar_mod_layer_call_fn_3244336

inputs
unknown:	
	unknown_0:

	unknown_1:	
	unknown_2:	

	unknown_3:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_ar_mod_layer_call_and_return_conditional_losses_3243979s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ
: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ÇK

D__inference_lstm_74_layer_call_and_return_conditional_losses_3244190

inputs>
+lstm_cell_74_matmul_readvariableop_resource:	A
-lstm_cell_74_matmul_1_readvariableop_resource:
;
,lstm_cell_74_biasadd_readvariableop_resource:	
identity¢#lstm_cell_74/BiasAdd/ReadVariableOp¢"lstm_cell_74/MatMul/ReadVariableOp¢$lstm_cell_74/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_74/MatMul/ReadVariableOpReadVariableOp+lstm_cell_74_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_74/MatMulMatMulstrided_slice_2:output:0*lstm_cell_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_74/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_74_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_74/MatMul_1MatMulzeros:output:0,lstm_cell_74/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_74/addAddV2lstm_cell_74/MatMul:product:0lstm_cell_74/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_74/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_74_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_74/BiasAddBiasAddlstm_cell_74/add:z:0+lstm_cell_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_74/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_74/splitSplit%lstm_cell_74/split/split_dim:output:0lstm_cell_74/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_74/SigmoidSigmoidlstm_cell_74/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_74/Sigmoid_1Sigmoidlstm_cell_74/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_74/mulMullstm_cell_74/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_74/ReluRelulstm_cell_74/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_74/mul_1Mullstm_cell_74/Sigmoid:y:0lstm_cell_74/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_74/add_1AddV2lstm_cell_74/mul:z:0lstm_cell_74/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_74/Sigmoid_2Sigmoidlstm_cell_74/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_74/Relu_1Relulstm_cell_74/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_74/mul_2Mullstm_cell_74/Sigmoid_2:y:0!lstm_cell_74/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_74_matmul_readvariableop_resource-lstm_cell_74_matmul_1_readvariableop_resource,lstm_cell_74_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3244105*
condR
while_cond_3244104*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_74/BiasAdd/ReadVariableOp#^lstm_cell_74/MatMul/ReadVariableOp%^lstm_cell_74/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 2J
#lstm_cell_74/BiasAdd/ReadVariableOp#lstm_cell_74/BiasAdd/ReadVariableOp2H
"lstm_cell_74/MatMul/ReadVariableOp"lstm_cell_74/MatMul/ReadVariableOp2L
$lstm_cell_74/MatMul_1/ReadVariableOp$lstm_cell_74/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¥
G
+__inference_dropout_7_layer_call_fn_3245297

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_3243950a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨K
³
!ar_mod_lstm_74_while_body_3243337:
6ar_mod_lstm_74_while_ar_mod_lstm_74_while_loop_counter@
<ar_mod_lstm_74_while_ar_mod_lstm_74_while_maximum_iterations$
 ar_mod_lstm_74_while_placeholder&
"ar_mod_lstm_74_while_placeholder_1&
"ar_mod_lstm_74_while_placeholder_2&
"ar_mod_lstm_74_while_placeholder_39
5ar_mod_lstm_74_while_ar_mod_lstm_74_strided_slice_1_0u
qar_mod_lstm_74_while_tensorarrayv2read_tensorlistgetitem_ar_mod_lstm_74_tensorarrayunstack_tensorlistfromtensor_0U
Bar_mod_lstm_74_while_lstm_cell_74_matmul_readvariableop_resource_0:	X
Dar_mod_lstm_74_while_lstm_cell_74_matmul_1_readvariableop_resource_0:
R
Car_mod_lstm_74_while_lstm_cell_74_biasadd_readvariableop_resource_0:	!
ar_mod_lstm_74_while_identity#
ar_mod_lstm_74_while_identity_1#
ar_mod_lstm_74_while_identity_2#
ar_mod_lstm_74_while_identity_3#
ar_mod_lstm_74_while_identity_4#
ar_mod_lstm_74_while_identity_57
3ar_mod_lstm_74_while_ar_mod_lstm_74_strided_slice_1s
oar_mod_lstm_74_while_tensorarrayv2read_tensorlistgetitem_ar_mod_lstm_74_tensorarrayunstack_tensorlistfromtensorS
@ar_mod_lstm_74_while_lstm_cell_74_matmul_readvariableop_resource:	V
Bar_mod_lstm_74_while_lstm_cell_74_matmul_1_readvariableop_resource:
P
Aar_mod_lstm_74_while_lstm_cell_74_biasadd_readvariableop_resource:	¢8ar_mod/lstm_74/while/lstm_cell_74/BiasAdd/ReadVariableOp¢7ar_mod/lstm_74/while/lstm_cell_74/MatMul/ReadVariableOp¢9ar_mod/lstm_74/while/lstm_cell_74/MatMul_1/ReadVariableOp
Far_mod/lstm_74/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ñ
8ar_mod/lstm_74/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqar_mod_lstm_74_while_tensorarrayv2read_tensorlistgetitem_ar_mod_lstm_74_tensorarrayunstack_tensorlistfromtensor_0 ar_mod_lstm_74_while_placeholderOar_mod/lstm_74/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0»
7ar_mod/lstm_74/while/lstm_cell_74/MatMul/ReadVariableOpReadVariableOpBar_mod_lstm_74_while_lstm_cell_74_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0ç
(ar_mod/lstm_74/while/lstm_cell_74/MatMulMatMul?ar_mod/lstm_74/while/TensorArrayV2Read/TensorListGetItem:item:0?ar_mod/lstm_74/while/lstm_cell_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
9ar_mod/lstm_74/while/lstm_cell_74/MatMul_1/ReadVariableOpReadVariableOpDar_mod_lstm_74_while_lstm_cell_74_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Î
*ar_mod/lstm_74/while/lstm_cell_74/MatMul_1MatMul"ar_mod_lstm_74_while_placeholder_2Aar_mod/lstm_74/while/lstm_cell_74/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
%ar_mod/lstm_74/while/lstm_cell_74/addAddV22ar_mod/lstm_74/while/lstm_cell_74/MatMul:product:04ar_mod/lstm_74/while/lstm_cell_74/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
8ar_mod/lstm_74/while/lstm_cell_74/BiasAdd/ReadVariableOpReadVariableOpCar_mod_lstm_74_while_lstm_cell_74_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ô
)ar_mod/lstm_74/while/lstm_cell_74/BiasAddBiasAdd)ar_mod/lstm_74/while/lstm_cell_74/add:z:0@ar_mod/lstm_74/while/lstm_cell_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
1ar_mod/lstm_74/while/lstm_cell_74/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
'ar_mod/lstm_74/while/lstm_cell_74/splitSplit:ar_mod/lstm_74/while/lstm_cell_74/split/split_dim:output:02ar_mod/lstm_74/while/lstm_cell_74/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
)ar_mod/lstm_74/while/lstm_cell_74/SigmoidSigmoid0ar_mod/lstm_74/while/lstm_cell_74/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+ar_mod/lstm_74/while/lstm_cell_74/Sigmoid_1Sigmoid0ar_mod/lstm_74/while/lstm_cell_74/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
%ar_mod/lstm_74/while/lstm_cell_74/mulMul/ar_mod/lstm_74/while/lstm_cell_74/Sigmoid_1:y:0"ar_mod_lstm_74_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&ar_mod/lstm_74/while/lstm_cell_74/ReluRelu0ar_mod/lstm_74/while/lstm_cell_74/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
'ar_mod/lstm_74/while/lstm_cell_74/mul_1Mul-ar_mod/lstm_74/while/lstm_cell_74/Sigmoid:y:04ar_mod/lstm_74/while/lstm_cell_74/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
'ar_mod/lstm_74/while/lstm_cell_74/add_1AddV2)ar_mod/lstm_74/while/lstm_cell_74/mul:z:0+ar_mod/lstm_74/while/lstm_cell_74/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+ar_mod/lstm_74/while/lstm_cell_74/Sigmoid_2Sigmoid0ar_mod/lstm_74/while/lstm_cell_74/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(ar_mod/lstm_74/while/lstm_cell_74/Relu_1Relu+ar_mod/lstm_74/while/lstm_cell_74/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
'ar_mod/lstm_74/while/lstm_cell_74/mul_2Mul/ar_mod/lstm_74/while/lstm_cell_74/Sigmoid_2:y:06ar_mod/lstm_74/while/lstm_cell_74/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
?ar_mod/lstm_74/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ©
9ar_mod/lstm_74/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"ar_mod_lstm_74_while_placeholder_1Har_mod/lstm_74/while/TensorArrayV2Write/TensorListSetItem/index:output:0+ar_mod/lstm_74/while/lstm_cell_74/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒ\
ar_mod/lstm_74/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
ar_mod/lstm_74/while/addAddV2 ar_mod_lstm_74_while_placeholder#ar_mod/lstm_74/while/add/y:output:0*
T0*
_output_shapes
: ^
ar_mod/lstm_74/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :£
ar_mod/lstm_74/while/add_1AddV26ar_mod_lstm_74_while_ar_mod_lstm_74_while_loop_counter%ar_mod/lstm_74/while/add_1/y:output:0*
T0*
_output_shapes
: 
ar_mod/lstm_74/while/IdentityIdentityar_mod/lstm_74/while/add_1:z:0^ar_mod/lstm_74/while/NoOp*
T0*
_output_shapes
: ¦
ar_mod/lstm_74/while/Identity_1Identity<ar_mod_lstm_74_while_ar_mod_lstm_74_while_maximum_iterations^ar_mod/lstm_74/while/NoOp*
T0*
_output_shapes
: 
ar_mod/lstm_74/while/Identity_2Identityar_mod/lstm_74/while/add:z:0^ar_mod/lstm_74/while/NoOp*
T0*
_output_shapes
: ³
ar_mod/lstm_74/while/Identity_3IdentityIar_mod/lstm_74/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^ar_mod/lstm_74/while/NoOp*
T0*
_output_shapes
: §
ar_mod/lstm_74/while/Identity_4Identity+ar_mod/lstm_74/while/lstm_cell_74/mul_2:z:0^ar_mod/lstm_74/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
ar_mod/lstm_74/while/Identity_5Identity+ar_mod/lstm_74/while/lstm_cell_74/add_1:z:0^ar_mod/lstm_74/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ar_mod/lstm_74/while/NoOpNoOp9^ar_mod/lstm_74/while/lstm_cell_74/BiasAdd/ReadVariableOp8^ar_mod/lstm_74/while/lstm_cell_74/MatMul/ReadVariableOp:^ar_mod/lstm_74/while/lstm_cell_74/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "l
3ar_mod_lstm_74_while_ar_mod_lstm_74_strided_slice_15ar_mod_lstm_74_while_ar_mod_lstm_74_strided_slice_1_0"G
ar_mod_lstm_74_while_identity&ar_mod/lstm_74/while/Identity:output:0"K
ar_mod_lstm_74_while_identity_1(ar_mod/lstm_74/while/Identity_1:output:0"K
ar_mod_lstm_74_while_identity_2(ar_mod/lstm_74/while/Identity_2:output:0"K
ar_mod_lstm_74_while_identity_3(ar_mod/lstm_74/while/Identity_3:output:0"K
ar_mod_lstm_74_while_identity_4(ar_mod/lstm_74/while/Identity_4:output:0"K
ar_mod_lstm_74_while_identity_5(ar_mod/lstm_74/while/Identity_5:output:0"
Aar_mod_lstm_74_while_lstm_cell_74_biasadd_readvariableop_resourceCar_mod_lstm_74_while_lstm_cell_74_biasadd_readvariableop_resource_0"
Bar_mod_lstm_74_while_lstm_cell_74_matmul_1_readvariableop_resourceDar_mod_lstm_74_while_lstm_cell_74_matmul_1_readvariableop_resource_0"
@ar_mod_lstm_74_while_lstm_cell_74_matmul_readvariableop_resourceBar_mod_lstm_74_while_lstm_cell_74_matmul_readvariableop_resource_0"ä
oar_mod_lstm_74_while_tensorarrayv2read_tensorlistgetitem_ar_mod_lstm_74_tensorarrayunstack_tensorlistfromtensorqar_mod_lstm_74_while_tensorarrayv2read_tensorlistgetitem_ar_mod_lstm_74_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2t
8ar_mod/lstm_74/while/lstm_cell_74/BiasAdd/ReadVariableOp8ar_mod/lstm_74/while/lstm_cell_74/BiasAdd/ReadVariableOp2r
7ar_mod/lstm_74/while/lstm_cell_74/MatMul/ReadVariableOp7ar_mod/lstm_74/while/lstm_cell_74/MatMul/ReadVariableOp2v
9ar_mod/lstm_74/while/lstm_cell_74/MatMul_1/ReadVariableOp9ar_mod/lstm_74/while/lstm_cell_74/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 


è
lstm_74_while_cond_3244410,
(lstm_74_while_lstm_74_while_loop_counter2
.lstm_74_while_lstm_74_while_maximum_iterations
lstm_74_while_placeholder
lstm_74_while_placeholder_1
lstm_74_while_placeholder_2
lstm_74_while_placeholder_3.
*lstm_74_while_less_lstm_74_strided_slice_1E
Alstm_74_while_lstm_74_while_cond_3244410___redundant_placeholder0E
Alstm_74_while_lstm_74_while_cond_3244410___redundant_placeholder1E
Alstm_74_while_lstm_74_while_cond_3244410___redundant_placeholder2E
Alstm_74_while_lstm_74_while_cond_3244410___redundant_placeholder3
lstm_74_while_identity

lstm_74/while/LessLesslstm_74_while_placeholder*lstm_74_while_less_lstm_74_strided_slice_1*
T0*
_output_shapes
: [
lstm_74/while/IdentityIdentitylstm_74/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_74_while_identitylstm_74/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ÇK

D__inference_lstm_74_layer_call_and_return_conditional_losses_3245292

inputs>
+lstm_cell_74_matmul_readvariableop_resource:	A
-lstm_cell_74_matmul_1_readvariableop_resource:
;
,lstm_cell_74_biasadd_readvariableop_resource:	
identity¢#lstm_cell_74/BiasAdd/ReadVariableOp¢"lstm_cell_74/MatMul/ReadVariableOp¢$lstm_cell_74/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_74/MatMul/ReadVariableOpReadVariableOp+lstm_cell_74_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_74/MatMulMatMulstrided_slice_2:output:0*lstm_cell_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_74/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_74_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_74/MatMul_1MatMulzeros:output:0,lstm_cell_74/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_74/addAddV2lstm_cell_74/MatMul:product:0lstm_cell_74/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_74/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_74_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_74/BiasAddBiasAddlstm_cell_74/add:z:0+lstm_cell_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_74/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_74/splitSplit%lstm_cell_74/split/split_dim:output:0lstm_cell_74/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_74/SigmoidSigmoidlstm_cell_74/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_74/Sigmoid_1Sigmoidlstm_cell_74/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_74/mulMullstm_cell_74/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_74/ReluRelulstm_cell_74/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_74/mul_1Mullstm_cell_74/Sigmoid:y:0lstm_cell_74/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_74/add_1AddV2lstm_cell_74/mul:z:0lstm_cell_74/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_74/Sigmoid_2Sigmoidlstm_cell_74/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_74/Relu_1Relulstm_cell_74/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_74/mul_2Mullstm_cell_74/Sigmoid_2:y:0!lstm_cell_74/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_74_matmul_readvariableop_resource-lstm_cell_74_matmul_1_readvariableop_resource,lstm_cell_74_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3245207*
condR
while_cond_3245206*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_74/BiasAdd/ReadVariableOp#^lstm_cell_74/MatMul/ReadVariableOp%^lstm_cell_74/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 2J
#lstm_cell_74/BiasAdd/ReadVariableOp#lstm_cell_74/BiasAdd/ReadVariableOp2H
"lstm_cell_74/MatMul/ReadVariableOp"lstm_cell_74/MatMul/ReadVariableOp2L
$lstm_cell_74/MatMul_1/ReadVariableOp$lstm_cell_74/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
9

D__inference_lstm_74_layer_call_and_return_conditional_losses_3243584

inputs'
lstm_cell_74_3243500:	(
lstm_cell_74_3243502:
#
lstm_cell_74_3243504:	
identity¢$lstm_cell_74/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskü
$lstm_cell_74/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_74_3243500lstm_cell_74_3243502lstm_cell_74_3243504*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_74_layer_call_and_return_conditional_losses_3243499n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : À
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_74_3243500lstm_cell_74_3243502lstm_cell_74_3243504*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3243514*
condR
while_cond_3243513*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
NoOpNoOp%^lstm_cell_74/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_74/StatefulPartitionedCall$lstm_cell_74/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì	
÷
E__inference_dense_71_layer_call_and_return_conditional_losses_3243962

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
z
N__inference_weighted_layer_32_layer_call_and_return_conditional_losses_3245352
inputs_0
inputs_1
identity^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   f
ReshapeReshapeinputs_1Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
MulMulinputs_0Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
S
IdentityIdentityMul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/1
ê
û
C__inference_ar_mod_layer_call_and_return_conditional_losses_3244298
input_47"
lstm_74_3244283:	#
lstm_74_3244285:

lstm_74_3244287:	#
dense_71_3244291:	

dense_71_3244293:

identity¢ dense_71/StatefulPartitionedCall¢!dropout_7/StatefulPartitionedCall¢lstm_74/StatefulPartitionedCall
lstm_74/StatefulPartitionedCallStatefulPartitionedCallinput_47lstm_74_3244283lstm_74_3244285lstm_74_3244287*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_74_layer_call_and_return_conditional_losses_3244190î
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(lstm_74/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_3244029
 dense_71/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0dense_71_3244291dense_71_3244293*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_71_layer_call_and_return_conditional_losses_3243962ý
!weighted_layer_32/PartitionedCallPartitionedCallinput_47)dense_71/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_weighted_layer_32_layer_call_and_return_conditional_losses_3243976}
IdentityIdentity*weighted_layer_32/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¯
NoOpNoOp!^dense_71/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall ^lstm_74/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ
: : : : : 2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2B
lstm_74/StatefulPartitionedCalllstm_74/StatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
input_47
Õ
í
(__inference_ar_mod_layer_call_fn_3244351

inputs
unknown:	
	unknown_0:

	unknown_1:	
	unknown_2:	

	unknown_3:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_ar_mod_layer_call_and_return_conditional_losses_3244234s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ
: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ÇK

D__inference_lstm_74_layer_call_and_return_conditional_losses_3245147

inputs>
+lstm_cell_74_matmul_readvariableop_resource:	A
-lstm_cell_74_matmul_1_readvariableop_resource:
;
,lstm_cell_74_biasadd_readvariableop_resource:	
identity¢#lstm_cell_74/BiasAdd/ReadVariableOp¢"lstm_cell_74/MatMul/ReadVariableOp¢$lstm_cell_74/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_74/MatMul/ReadVariableOpReadVariableOp+lstm_cell_74_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_74/MatMulMatMulstrided_slice_2:output:0*lstm_cell_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_74/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_74_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_74/MatMul_1MatMulzeros:output:0,lstm_cell_74/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_74/addAddV2lstm_cell_74/MatMul:product:0lstm_cell_74/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_74/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_74_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_74/BiasAddBiasAddlstm_cell_74/add:z:0+lstm_cell_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_74/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_74/splitSplit%lstm_cell_74/split/split_dim:output:0lstm_cell_74/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_74/SigmoidSigmoidlstm_cell_74/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_74/Sigmoid_1Sigmoidlstm_cell_74/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_74/mulMullstm_cell_74/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_74/ReluRelulstm_cell_74/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_74/mul_1Mullstm_cell_74/Sigmoid:y:0lstm_cell_74/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_74/add_1AddV2lstm_cell_74/mul:z:0lstm_cell_74/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_74/Sigmoid_2Sigmoidlstm_cell_74/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_74/Relu_1Relulstm_cell_74/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_74/mul_2Mullstm_cell_74/Sigmoid_2:y:0!lstm_cell_74/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_74_matmul_readvariableop_resource-lstm_cell_74_matmul_1_readvariableop_resource,lstm_cell_74_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3245062*
condR
while_cond_3245061*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_74/BiasAdd/ReadVariableOp#^lstm_cell_74/MatMul/ReadVariableOp%^lstm_cell_74/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 2J
#lstm_cell_74/BiasAdd/ReadVariableOp#lstm_cell_74/BiasAdd/ReadVariableOp2H
"lstm_cell_74/MatMul/ReadVariableOp"lstm_cell_74/MatMul/ReadVariableOp2L
$lstm_cell_74/MatMul_1/ReadVariableOp$lstm_cell_74/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs


e
F__inference_dropout_7_layer_call_and_return_conditional_losses_3245319

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seedÒ	[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì

I__inference_lstm_cell_74_layer_call_and_return_conditional_losses_3243499

inputs

states
states_11
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
¾
È
while_cond_3243513
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3243513___redundant_placeholder05
1while_while_cond_3243513___redundant_placeholder15
1while_while_cond_3243513___redundant_placeholder25
1while_while_cond_3243513___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
êK

D__inference_lstm_74_layer_call_and_return_conditional_losses_3244857
inputs_0>
+lstm_cell_74_matmul_readvariableop_resource:	A
-lstm_cell_74_matmul_1_readvariableop_resource:
;
,lstm_cell_74_biasadd_readvariableop_resource:	
identity¢#lstm_cell_74/BiasAdd/ReadVariableOp¢"lstm_cell_74/MatMul/ReadVariableOp¢$lstm_cell_74/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_74/MatMul/ReadVariableOpReadVariableOp+lstm_cell_74_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_74/MatMulMatMulstrided_slice_2:output:0*lstm_cell_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_74/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_74_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_74/MatMul_1MatMulzeros:output:0,lstm_cell_74/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_74/addAddV2lstm_cell_74/MatMul:product:0lstm_cell_74/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_74/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_74_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_74/BiasAddBiasAddlstm_cell_74/add:z:0+lstm_cell_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_74/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_74/splitSplit%lstm_cell_74/split/split_dim:output:0lstm_cell_74/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_74/SigmoidSigmoidlstm_cell_74/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_74/Sigmoid_1Sigmoidlstm_cell_74/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_74/mulMullstm_cell_74/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_74/ReluRelulstm_cell_74/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_74/mul_1Mullstm_cell_74/Sigmoid:y:0lstm_cell_74/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_74/add_1AddV2lstm_cell_74/mul:z:0lstm_cell_74/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_74/Sigmoid_2Sigmoidlstm_cell_74/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_74/Relu_1Relulstm_cell_74/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_74/mul_2Mullstm_cell_74/Sigmoid_2:y:0!lstm_cell_74/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_74_matmul_readvariableop_resource-lstm_cell_74_matmul_1_readvariableop_resource,lstm_cell_74_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3244772*
condR
while_cond_3244771*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_74/BiasAdd/ReadVariableOp#^lstm_cell_74/MatMul/ReadVariableOp%^lstm_cell_74/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_74/BiasAdd/ReadVariableOp#lstm_cell_74/BiasAdd/ReadVariableOp2H
"lstm_cell_74/MatMul/ReadVariableOp"lstm_cell_74/MatMul/ReadVariableOp2L
$lstm_cell_74/MatMul_1/ReadVariableOp$lstm_cell_74/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ü
·
)__inference_lstm_74_layer_call_fn_3244712

inputs
unknown:	
	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_74_layer_call_and_return_conditional_losses_3244190p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¾
È
while_cond_3244771
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3244771___redundant_placeholder05
1while_while_cond_3244771___redundant_placeholder15
1while_while_cond_3244771___redundant_placeholder25
1while_while_cond_3244771___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¾
È
while_cond_3244916
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3244916___redundant_placeholder05
1while_while_cond_3244916___redundant_placeholder15
1while_while_cond_3244916___redundant_placeholder25
1while_while_cond_3244916___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¾
È
while_cond_3244104
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3244104___redundant_placeholder05
1while_while_cond_3244104___redundant_placeholder15
1while_while_cond_3244104___redundant_placeholder25
1while_while_cond_3244104___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Â
×
C__inference_ar_mod_layer_call_and_return_conditional_losses_3244280
input_47"
lstm_74_3244265:	#
lstm_74_3244267:

lstm_74_3244269:	#
dense_71_3244273:	

dense_71_3244275:

identity¢ dense_71/StatefulPartitionedCall¢lstm_74/StatefulPartitionedCall
lstm_74/StatefulPartitionedCallStatefulPartitionedCallinput_47lstm_74_3244265lstm_74_3244267lstm_74_3244269*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_74_layer_call_and_return_conditional_losses_3243937Þ
dropout_7/PartitionedCallPartitionedCall(lstm_74/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_3243950
 dense_71/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0dense_71_3244273dense_71_3244275*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_71_layer_call_and_return_conditional_losses_3243962ý
!weighted_layer_32/PartitionedCallPartitionedCallinput_47)dense_71/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_weighted_layer_32_layer_call_and_return_conditional_losses_3243976}
IdentityIdentity*weighted_layer_32/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp!^dense_71/StatefulPartitionedCall ^lstm_74/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ
: : : : : 2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall2B
lstm_74/StatefulPartitionedCalllstm_74/StatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
input_47
¡
x
N__inference_weighted_layer_32_layer_call_and_return_conditional_losses_3243976

inputs
inputs_1
identity^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   f
ReshapeReshapeinputs_1Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
MulMulinputsReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
S
IdentityIdentityMul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs


e
F__inference_dropout_7_layer_call_and_return_conditional_losses_3244029

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seedÒ	[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿9
Ó
while_body_3244772
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_74_matmul_readvariableop_resource_0:	I
5while_lstm_cell_74_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_74_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_74_matmul_readvariableop_resource:	G
3while_lstm_cell_74_matmul_1_readvariableop_resource:
A
2while_lstm_cell_74_biasadd_readvariableop_resource:	¢)while/lstm_cell_74/BiasAdd/ReadVariableOp¢(while/lstm_cell_74/MatMul/ReadVariableOp¢*while/lstm_cell_74/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_74/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_74_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_74/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_74/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_74_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_74/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_74/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/addAddV2#while/lstm_cell_74/MatMul:product:0%while/lstm_cell_74/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_74/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_74_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_74/BiasAddBiasAddwhile/lstm_cell_74/add:z:01while/lstm_cell_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_74/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_74/splitSplit+while/lstm_cell_74/split/split_dim:output:0#while/lstm_cell_74/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_74/SigmoidSigmoid!while/lstm_cell_74/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_74/Sigmoid_1Sigmoid!while/lstm_cell_74/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/mulMul while/lstm_cell_74/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_74/ReluRelu!while/lstm_cell_74/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/mul_1Mulwhile/lstm_cell_74/Sigmoid:y:0%while/lstm_cell_74/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/add_1AddV2while/lstm_cell_74/mul:z:0while/lstm_cell_74/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_74/Sigmoid_2Sigmoid!while/lstm_cell_74/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_74/Relu_1Reluwhile/lstm_cell_74/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/mul_2Mul while/lstm_cell_74/Sigmoid_2:y:0'while/lstm_cell_74/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_74/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_74/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_74/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_74/BiasAdd/ReadVariableOp)^while/lstm_cell_74/MatMul/ReadVariableOp+^while/lstm_cell_74/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_74_biasadd_readvariableop_resource4while_lstm_cell_74_biasadd_readvariableop_resource_0"l
3while_lstm_cell_74_matmul_1_readvariableop_resource5while_lstm_cell_74_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_74_matmul_readvariableop_resource3while_lstm_cell_74_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_74/BiasAdd/ReadVariableOp)while/lstm_cell_74/BiasAdd/ReadVariableOp2T
(while/lstm_cell_74/MatMul/ReadVariableOp(while/lstm_cell_74/MatMul/ReadVariableOp2X
*while/lstm_cell_74/MatMul_1/ReadVariableOp*while/lstm_cell_74/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ø
ø
.__inference_lstm_cell_74_layer_call_fn_3245386

inputs
states_0
states_1
unknown:	
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_74_layer_call_and_return_conditional_losses_3243647p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ø
ø
.__inference_lstm_cell_74_layer_call_fn_3245369

inputs
states_0
states_1
unknown:	
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_74_layer_call_and_return_conditional_losses_3243499p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
¿9
Ó
while_body_3244917
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_74_matmul_readvariableop_resource_0:	I
5while_lstm_cell_74_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_74_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_74_matmul_readvariableop_resource:	G
3while_lstm_cell_74_matmul_1_readvariableop_resource:
A
2while_lstm_cell_74_biasadd_readvariableop_resource:	¢)while/lstm_cell_74/BiasAdd/ReadVariableOp¢(while/lstm_cell_74/MatMul/ReadVariableOp¢*while/lstm_cell_74/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_74/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_74_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_74/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_74/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_74_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_74/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_74/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/addAddV2#while/lstm_cell_74/MatMul:product:0%while/lstm_cell_74/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_74/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_74_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_74/BiasAddBiasAddwhile/lstm_cell_74/add:z:01while/lstm_cell_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_74/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_74/splitSplit+while/lstm_cell_74/split/split_dim:output:0#while/lstm_cell_74/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_74/SigmoidSigmoid!while/lstm_cell_74/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_74/Sigmoid_1Sigmoid!while/lstm_cell_74/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/mulMul while/lstm_cell_74/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_74/ReluRelu!while/lstm_cell_74/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/mul_1Mulwhile/lstm_cell_74/Sigmoid:y:0%while/lstm_cell_74/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/add_1AddV2while/lstm_cell_74/mul:z:0while/lstm_cell_74/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_74/Sigmoid_2Sigmoid!while/lstm_cell_74/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_74/Relu_1Reluwhile/lstm_cell_74/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_74/mul_2Mul while/lstm_cell_74/Sigmoid_2:y:0'while/lstm_cell_74/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_74/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_74/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_74/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_74/BiasAdd/ReadVariableOp)^while/lstm_cell_74/MatMul/ReadVariableOp+^while/lstm_cell_74/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_74_biasadd_readvariableop_resource4while_lstm_cell_74_biasadd_readvariableop_resource_0"l
3while_lstm_cell_74_matmul_1_readvariableop_resource5while_lstm_cell_74_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_74_matmul_readvariableop_resource3while_lstm_cell_74_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_74/BiasAdd/ReadVariableOp)while/lstm_cell_74/BiasAdd/ReadVariableOp2T
(while/lstm_cell_74/MatMul/ReadVariableOp(while/lstm_cell_74/MatMul/ReadVariableOp2X
*while/lstm_cell_74/MatMul_1/ReadVariableOp*while/lstm_cell_74/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
º5
­

 __inference__traced_save_3245539
file_prefix.
*savev2_dense_71_kernel_read_readvariableop,
(savev2_dense_71_bias_read_readvariableop:
6savev2_lstm_74_lstm_cell_74_kernel_read_readvariableopD
@savev2_lstm_74_lstm_cell_74_recurrent_kernel_read_readvariableop8
4savev2_lstm_74_lstm_cell_74_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_71_kernel_m_read_readvariableop3
/savev2_adam_dense_71_bias_m_read_readvariableopA
=savev2_adam_lstm_74_lstm_cell_74_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_74_lstm_cell_74_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_74_lstm_cell_74_bias_m_read_readvariableop5
1savev2_adam_dense_71_kernel_v_read_readvariableop3
/savev2_adam_dense_71_bias_v_read_readvariableopA
=savev2_adam_lstm_74_lstm_cell_74_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_74_lstm_cell_74_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_74_lstm_cell_74_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ¥
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Î

valueÄ
BÁ
B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B ®

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_71_kernel_read_readvariableop(savev2_dense_71_bias_read_readvariableop6savev2_lstm_74_lstm_cell_74_kernel_read_readvariableop@savev2_lstm_74_lstm_cell_74_recurrent_kernel_read_readvariableop4savev2_lstm_74_lstm_cell_74_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_71_kernel_m_read_readvariableop/savev2_adam_dense_71_bias_m_read_readvariableop=savev2_adam_lstm_74_lstm_cell_74_kernel_m_read_readvariableopGsavev2_adam_lstm_74_lstm_cell_74_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_74_lstm_cell_74_bias_m_read_readvariableop1savev2_adam_dense_71_kernel_v_read_readvariableop/savev2_adam_dense_71_bias_v_read_readvariableop=savev2_adam_lstm_74_lstm_cell_74_kernel_v_read_readvariableopGsavev2_adam_lstm_74_lstm_cell_74_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_74_lstm_cell_74_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*´
_input_shapes¢
: :	
:
:	:
:: : : : : : : :	
:
:	:
::	
:
:	:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	
: 

_output_shapes
:
:%!

_output_shapes
:	:&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	
: 

_output_shapes
:
:%!

_output_shapes
:	:&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:%!

_output_shapes
:	:&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: 
º
Õ
C__inference_ar_mod_layer_call_and_return_conditional_losses_3243979

inputs"
lstm_74_3243938:	#
lstm_74_3243940:

lstm_74_3243942:	#
dense_71_3243963:	

dense_71_3243965:

identity¢ dense_71/StatefulPartitionedCall¢lstm_74/StatefulPartitionedCall
lstm_74/StatefulPartitionedCallStatefulPartitionedCallinputslstm_74_3243938lstm_74_3243940lstm_74_3243942*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_74_layer_call_and_return_conditional_losses_3243937Þ
dropout_7/PartitionedCallPartitionedCall(lstm_74/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_3243950
 dense_71/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0dense_71_3243963dense_71_3243965*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_71_layer_call_and_return_conditional_losses_3243962û
!weighted_layer_32/PartitionedCallPartitionedCallinputs)dense_71/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_weighted_layer_32_layer_call_and_return_conditional_losses_3243976}
IdentityIdentity*weighted_layer_32/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp!^dense_71/StatefulPartitionedCall ^lstm_74/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ
: : : : : 2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall2B
lstm_74/StatefulPartitionedCalllstm_74/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Û
ï
(__inference_ar_mod_layer_call_fn_3243992
input_47
unknown:	
	unknown_0:

	unknown_1:	
	unknown_2:	

	unknown_3:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_47unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_ar_mod_layer_call_and_return_conditional_losses_3243979s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ
: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
input_47
¾
È
while_cond_3243706
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3243706___redundant_placeholder05
1while_while_cond_3243706___redundant_placeholder15
1while_while_cond_3243706___redundant_placeholder25
1while_while_cond_3243706___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¾
È
while_cond_3245061
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3245061___redundant_placeholder05
1while_while_cond_3245061___redundant_placeholder15
1while_while_cond_3245061___redundant_placeholder25
1while_while_cond_3245061___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
$
ì
while_body_3243514
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_74_3243538_0:	0
while_lstm_cell_74_3243540_0:
+
while_lstm_cell_74_3243542_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_74_3243538:	.
while_lstm_cell_74_3243540:
)
while_lstm_cell_74_3243542:	¢*while/lstm_cell_74/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0º
*while/lstm_cell_74/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_74_3243538_0while_lstm_cell_74_3243540_0while_lstm_cell_74_3243542_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_74_layer_call_and_return_conditional_losses_3243499r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_74/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity3while/lstm_cell_74/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity3while/lstm_cell_74/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

while/NoOpNoOp+^while/lstm_cell_74/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_74_3243538while_lstm_cell_74_3243538_0":
while_lstm_cell_74_3243540while_lstm_cell_74_3243540_0":
while_lstm_cell_74_3243542while_lstm_cell_74_3243542_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_74/StatefulPartitionedCall*while/lstm_cell_74/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
÷
d
+__inference_dropout_7_layer_call_fn_3245302

inputs
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_3244029p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
_
3__inference_weighted_layer_32_layer_call_fn_3245344
inputs_0
inputs_1
identityÊ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_weighted_layer_32_layer_call_and_return_conditional_losses_3243976d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/1
¾
È
while_cond_3245206
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3245206___redundant_placeholder05
1while_while_cond_3245206___redundant_placeholder15
1while_while_cond_3245206___redundant_placeholder25
1while_while_cond_3245206___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

¹
)__inference_lstm_74_layer_call_fn_3244690
inputs_0
unknown:	
	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_74_layer_call_and_return_conditional_losses_3243777p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
êK

D__inference_lstm_74_layer_call_and_return_conditional_losses_3245002
inputs_0>
+lstm_cell_74_matmul_readvariableop_resource:	A
-lstm_cell_74_matmul_1_readvariableop_resource:
;
,lstm_cell_74_biasadd_readvariableop_resource:	
identity¢#lstm_cell_74/BiasAdd/ReadVariableOp¢"lstm_cell_74/MatMul/ReadVariableOp¢$lstm_cell_74/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_74/MatMul/ReadVariableOpReadVariableOp+lstm_cell_74_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_74/MatMulMatMulstrided_slice_2:output:0*lstm_cell_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_74/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_74_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_74/MatMul_1MatMulzeros:output:0,lstm_cell_74/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_74/addAddV2lstm_cell_74/MatMul:product:0lstm_cell_74/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_74/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_74_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_74/BiasAddBiasAddlstm_cell_74/add:z:0+lstm_cell_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_74/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_74/splitSplit%lstm_cell_74/split/split_dim:output:0lstm_cell_74/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_74/SigmoidSigmoidlstm_cell_74/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_74/Sigmoid_1Sigmoidlstm_cell_74/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_74/mulMullstm_cell_74/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_74/ReluRelulstm_cell_74/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_74/mul_1Mullstm_cell_74/Sigmoid:y:0lstm_cell_74/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_74/add_1AddV2lstm_cell_74/mul:z:0lstm_cell_74/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_74/Sigmoid_2Sigmoidlstm_cell_74/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_74/Relu_1Relulstm_cell_74/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_74/mul_2Mullstm_cell_74/Sigmoid_2:y:0!lstm_cell_74/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_74_matmul_readvariableop_resource-lstm_cell_74_matmul_1_readvariableop_resource,lstm_cell_74_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3244917*
condR
while_cond_3244916*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_74/BiasAdd/ReadVariableOp#^lstm_cell_74/MatMul/ReadVariableOp%^lstm_cell_74/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_74/BiasAdd/ReadVariableOp#lstm_cell_74/BiasAdd/ReadVariableOp2H
"lstm_cell_74/MatMul/ReadVariableOp"lstm_cell_74/MatMul/ReadVariableOp2L
$lstm_cell_74/MatMul_1/ReadVariableOp$lstm_cell_74/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
â
ù
C__inference_ar_mod_layer_call_and_return_conditional_losses_3244234

inputs"
lstm_74_3244219:	#
lstm_74_3244221:

lstm_74_3244223:	#
dense_71_3244227:	

dense_71_3244229:

identity¢ dense_71/StatefulPartitionedCall¢!dropout_7/StatefulPartitionedCall¢lstm_74/StatefulPartitionedCall
lstm_74/StatefulPartitionedCallStatefulPartitionedCallinputslstm_74_3244219lstm_74_3244221lstm_74_3244223*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_74_layer_call_and_return_conditional_losses_3244190î
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(lstm_74/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_3244029
 dense_71/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0dense_71_3244227dense_71_3244229*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_71_layer_call_and_return_conditional_losses_3243962û
!weighted_layer_32/PartitionedCallPartitionedCallinputs)dense_71/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_weighted_layer_32_layer_call_and_return_conditional_losses_3243976}
IdentityIdentity*weighted_layer_32/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¯
NoOpNoOp!^dense_71/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall ^lstm_74/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ
: : : : : 2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2B
lstm_74/StatefulPartitionedCalllstm_74/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs"¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¾
serving_defaultª
A
input_475
serving_default_input_47:0ÿÿÿÿÿÿÿÿÿ
I
weighted_layer_324
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:Ã
Ø
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
Ú
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
¼
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
»
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
¥
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
C
-0
.1
/2
%3
&4"
trackable_list_wrapper
C
-0
.1
/2
%3
&4"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ö
5trace_0
6trace_1
7trace_2
8trace_32ë
(__inference_ar_mod_layer_call_fn_3243992
(__inference_ar_mod_layer_call_fn_3244336
(__inference_ar_mod_layer_call_fn_3244351
(__inference_ar_mod_layer_call_fn_3244262À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z5trace_0z6trace_1z7trace_2z8trace_3
Â
9trace_0
:trace_1
;trace_2
<trace_32×
C__inference_ar_mod_layer_call_and_return_conditional_losses_3244506
C__inference_ar_mod_layer_call_and_return_conditional_losses_3244668
C__inference_ar_mod_layer_call_and_return_conditional_losses_3244280
C__inference_ar_mod_layer_call_and_return_conditional_losses_3244298À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z9trace_0z:trace_1z;trace_2z<trace_3
ÎBË
"__inference__wrapped_model_3243432input_47"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
µ
=iter

>beta_1

?beta_2
	@decay
Alearning_rate%m~&m-m.m/m%v&v-v.v/v"
	optimizer
,
Bserving_default"
signature_map
5
-0
.1
/2"
trackable_list_wrapper
5
-0
.1
/2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

Cstates
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ï
Itrace_0
Jtrace_1
Ktrace_2
Ltrace_32
)__inference_lstm_74_layer_call_fn_3244679
)__inference_lstm_74_layer_call_fn_3244690
)__inference_lstm_74_layer_call_fn_3244701
)__inference_lstm_74_layer_call_fn_3244712Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zItrace_0zJtrace_1zKtrace_2zLtrace_3
Û
Mtrace_0
Ntrace_1
Otrace_2
Ptrace_32ð
D__inference_lstm_74_layer_call_and_return_conditional_losses_3244857
D__inference_lstm_74_layer_call_and_return_conditional_losses_3245002
D__inference_lstm_74_layer_call_and_return_conditional_losses_3245147
D__inference_lstm_74_layer_call_and_return_conditional_losses_3245292Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zMtrace_0zNtrace_1zOtrace_2zPtrace_3
"
_generic_user_object
ø
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
W_random_generator
X
state_size

-kernel
.recurrent_kernel
/bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
È
^trace_0
_trace_12
+__inference_dropout_7_layer_call_fn_3245297
+__inference_dropout_7_layer_call_fn_3245302´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z^trace_0z_trace_1
þ
`trace_0
atrace_12Ç
F__inference_dropout_7_layer_call_and_return_conditional_losses_3245307
F__inference_dropout_7_layer_call_and_return_conditional_losses_3245319´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z`trace_0zatrace_1
"
_generic_user_object
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
î
gtrace_02Ñ
*__inference_dense_71_layer_call_fn_3245328¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zgtrace_0

htrace_02ì
E__inference_dense_71_layer_call_and_return_conditional_losses_3245338¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zhtrace_0
": 	
2dense_71/kernel
:
2dense_71/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
÷
ntrace_02Ú
3__inference_weighted_layer_32_layer_call_fn_3245344¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zntrace_0

otrace_02õ
N__inference_weighted_layer_32_layer_call_and_return_conditional_losses_3245352¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zotrace_0
.:,	2lstm_74/lstm_cell_74/kernel
9:7
2%lstm_74/lstm_cell_74/recurrent_kernel
(:&2lstm_74/lstm_cell_74/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
'
p0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
üBù
(__inference_ar_mod_layer_call_fn_3243992input_47"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
úB÷
(__inference_ar_mod_layer_call_fn_3244336inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
úB÷
(__inference_ar_mod_layer_call_fn_3244351inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
üBù
(__inference_ar_mod_layer_call_fn_3244262input_47"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
C__inference_ar_mod_layer_call_and_return_conditional_losses_3244506inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
C__inference_ar_mod_layer_call_and_return_conditional_losses_3244668inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
C__inference_ar_mod_layer_call_and_return_conditional_losses_3244280input_47"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
C__inference_ar_mod_layer_call_and_return_conditional_losses_3244298input_47"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÍBÊ
%__inference_signature_wrapper_3244321input_47"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
)__inference_lstm_74_layer_call_fn_3244679inputs/0"Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
)__inference_lstm_74_layer_call_fn_3244690inputs/0"Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
)__inference_lstm_74_layer_call_fn_3244701inputs"Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
)__inference_lstm_74_layer_call_fn_3244712inputs"Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
­Bª
D__inference_lstm_74_layer_call_and_return_conditional_losses_3244857inputs/0"Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
­Bª
D__inference_lstm_74_layer_call_and_return_conditional_losses_3245002inputs/0"Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
«B¨
D__inference_lstm_74_layer_call_and_return_conditional_losses_3245147inputs"Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
«B¨
D__inference_lstm_74_layer_call_and_return_conditional_losses_3245292inputs"Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
5
-0
.1
/2"
trackable_list_wrapper
5
-0
.1
/2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
Ø
vtrace_0
wtrace_12¡
.__inference_lstm_cell_74_layer_call_fn_3245369
.__inference_lstm_cell_74_layer_call_fn_3245386¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zvtrace_0zwtrace_1

xtrace_0
ytrace_12×
I__inference_lstm_cell_74_layer_call_and_return_conditional_losses_3245418
I__inference_lstm_cell_74_layer_call_and_return_conditional_losses_3245450¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zxtrace_0zytrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ñBî
+__inference_dropout_7_layer_call_fn_3245297inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ñBî
+__inference_dropout_7_layer_call_fn_3245302inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
F__inference_dropout_7_layer_call_and_return_conditional_losses_3245307inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
F__inference_dropout_7_layer_call_and_return_conditional_losses_3245319inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_dense_71_layer_call_fn_3245328inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_dense_71_layer_call_and_return_conditional_losses_3245338inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
óBð
3__inference_weighted_layer_32_layer_call_fn_3245344inputs/0inputs/1"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
N__inference_weighted_layer_32_layer_call_and_return_conditional_losses_3245352inputs/0inputs/1"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
N
z	variables
{	keras_api
	|total
	}count"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_lstm_cell_74_layer_call_fn_3245369inputsstates/0states/1"¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
.__inference_lstm_cell_74_layer_call_fn_3245386inputsstates/0states/1"¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
­Bª
I__inference_lstm_cell_74_layer_call_and_return_conditional_losses_3245418inputsstates/0states/1"¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
­Bª
I__inference_lstm_cell_74_layer_call_and_return_conditional_losses_3245450inputsstates/0states/1"¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
.
|0
}1"
trackable_list_wrapper
-
z	variables"
_generic_user_object
:  (2total
:  (2count
':%	
2Adam/dense_71/kernel/m
 :
2Adam/dense_71/bias/m
3:1	2"Adam/lstm_74/lstm_cell_74/kernel/m
>:<
2,Adam/lstm_74/lstm_cell_74/recurrent_kernel/m
-:+2 Adam/lstm_74/lstm_cell_74/bias/m
':%	
2Adam/dense_71/kernel/v
 :
2Adam/dense_71/bias/v
3:1	2"Adam/lstm_74/lstm_cell_74/kernel/v
>:<
2,Adam/lstm_74/lstm_cell_74/recurrent_kernel/v
-:+2 Adam/lstm_74/lstm_cell_74/bias/v°
"__inference__wrapped_model_3243432-./%&5¢2
+¢(
&#
input_47ÿÿÿÿÿÿÿÿÿ

ª "IªF
D
weighted_layer_32/,
weighted_layer_32ÿÿÿÿÿÿÿÿÿ
¸
C__inference_ar_mod_layer_call_and_return_conditional_losses_3244280q-./%&=¢:
3¢0
&#
input_47ÿÿÿÿÿÿÿÿÿ

p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ

 ¸
C__inference_ar_mod_layer_call_and_return_conditional_losses_3244298q-./%&=¢:
3¢0
&#
input_47ÿÿÿÿÿÿÿÿÿ

p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ

 ¶
C__inference_ar_mod_layer_call_and_return_conditional_losses_3244506o-./%&;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ

p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ

 ¶
C__inference_ar_mod_layer_call_and_return_conditional_losses_3244668o-./%&;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ

p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ

 
(__inference_ar_mod_layer_call_fn_3243992d-./%&=¢:
3¢0
&#
input_47ÿÿÿÿÿÿÿÿÿ

p 

 
ª "ÿÿÿÿÿÿÿÿÿ

(__inference_ar_mod_layer_call_fn_3244262d-./%&=¢:
3¢0
&#
input_47ÿÿÿÿÿÿÿÿÿ

p

 
ª "ÿÿÿÿÿÿÿÿÿ

(__inference_ar_mod_layer_call_fn_3244336b-./%&;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ

p 

 
ª "ÿÿÿÿÿÿÿÿÿ

(__inference_ar_mod_layer_call_fn_3244351b-./%&;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ

p

 
ª "ÿÿÿÿÿÿÿÿÿ
¦
E__inference_dense_71_layer_call_and_return_conditional_losses_3245338]%&0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ~
*__inference_dense_71_layer_call_fn_3245328P%&0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
¨
F__inference_dropout_7_layer_call_and_return_conditional_losses_3245307^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¨
F__inference_dropout_7_layer_call_and_return_conditional_losses_3245319^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dropout_7_layer_call_fn_3245297Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_dropout_7_layer_call_fn_3245302Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÆ
D__inference_lstm_74_layer_call_and_return_conditional_losses_3244857~-./O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Æ
D__inference_lstm_74_layer_call_and_return_conditional_losses_3245002~-./O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¶
D__inference_lstm_74_layer_call_and_return_conditional_losses_3245147n-./?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ


 
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¶
D__inference_lstm_74_layer_call_and_return_conditional_losses_3245292n-./?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ


 
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_lstm_74_layer_call_fn_3244679q-./O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_lstm_74_layer_call_fn_3244690q-./O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_lstm_74_layer_call_fn_3244701a-./?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ


 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_lstm_74_layer_call_fn_3244712a-./?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ


 
p

 
ª "ÿÿÿÿÿÿÿÿÿÐ
I__inference_lstm_cell_74_layer_call_and_return_conditional_losses_3245418-./¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 Ð
I__inference_lstm_cell_74_layer_call_and_return_conditional_losses_3245450-./¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 ¥
.__inference_lstm_cell_74_layer_call_fn_3245369ò-./¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ¥
.__inference_lstm_cell_74_layer_call_fn_3245386ò-./¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ¿
%__inference_signature_wrapper_3244321-./%&A¢>
¢ 
7ª4
2
input_47&#
input_47ÿÿÿÿÿÿÿÿÿ
"IªF
D
weighted_layer_32/,
weighted_layer_32ÿÿÿÿÿÿÿÿÿ
Þ
N__inference_weighted_layer_32_layer_call_and_return_conditional_losses_3245352^¢[
T¢Q
OL
&#
inputs/0ÿÿÿÿÿÿÿÿÿ

"
inputs/1ÿÿÿÿÿÿÿÿÿ

ª ")¢&

0ÿÿÿÿÿÿÿÿÿ

 µ
3__inference_weighted_layer_32_layer_call_fn_3245344~^¢[
T¢Q
OL
&#
inputs/0ÿÿÿÿÿÿÿÿÿ

"
inputs/1ÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ
