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
 Adam/lstm_75/lstm_cell_75/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_75/lstm_cell_75/bias/v

4Adam/lstm_75/lstm_cell_75/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_75/lstm_cell_75/bias/v*
_output_shapes	
:*
dtype0
¶
,Adam/lstm_75/lstm_cell_75/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/lstm_75/lstm_cell_75/recurrent_kernel/v
¯
@Adam/lstm_75/lstm_cell_75/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_75/lstm_cell_75/recurrent_kernel/v* 
_output_shapes
:
*
dtype0
¡
"Adam/lstm_75/lstm_cell_75/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/lstm_75/lstm_cell_75/kernel/v

6Adam/lstm_75/lstm_cell_75/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_75/lstm_cell_75/kernel/v*
_output_shapes
:	*
dtype0

Adam/dense_72/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_72/bias/v
y
(Adam/dense_72/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_72/bias/v*
_output_shapes
:
*
dtype0

Adam/dense_72/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*'
shared_nameAdam/dense_72/kernel/v

*Adam/dense_72/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_72/kernel/v*
_output_shapes
:	
*
dtype0

 Adam/lstm_75/lstm_cell_75/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_75/lstm_cell_75/bias/m

4Adam/lstm_75/lstm_cell_75/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_75/lstm_cell_75/bias/m*
_output_shapes	
:*
dtype0
¶
,Adam/lstm_75/lstm_cell_75/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/lstm_75/lstm_cell_75/recurrent_kernel/m
¯
@Adam/lstm_75/lstm_cell_75/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_75/lstm_cell_75/recurrent_kernel/m* 
_output_shapes
:
*
dtype0
¡
"Adam/lstm_75/lstm_cell_75/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/lstm_75/lstm_cell_75/kernel/m

6Adam/lstm_75/lstm_cell_75/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_75/lstm_cell_75/kernel/m*
_output_shapes
:	*
dtype0

Adam/dense_72/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_72/bias/m
y
(Adam/dense_72/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_72/bias/m*
_output_shapes
:
*
dtype0

Adam/dense_72/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*'
shared_nameAdam/dense_72/kernel/m

*Adam/dense_72/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_72/kernel/m*
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
lstm_75/lstm_cell_75/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_75/lstm_cell_75/bias

-lstm_75/lstm_cell_75/bias/Read/ReadVariableOpReadVariableOplstm_75/lstm_cell_75/bias*
_output_shapes	
:*
dtype0
¨
%lstm_75/lstm_cell_75/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%lstm_75/lstm_cell_75/recurrent_kernel
¡
9lstm_75/lstm_cell_75/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_75/lstm_cell_75/recurrent_kernel* 
_output_shapes
:
*
dtype0

lstm_75/lstm_cell_75/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_namelstm_75/lstm_cell_75/kernel

/lstm_75/lstm_cell_75/kernel/Read/ReadVariableOpReadVariableOplstm_75/lstm_cell_75/kernel*
_output_shapes
:	*
dtype0
r
dense_72/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_72/bias
k
!dense_72/bias/Read/ReadVariableOpReadVariableOpdense_72/bias*
_output_shapes
:
*
dtype0
{
dense_72/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
* 
shared_namedense_72/kernel
t
#dense_72/kernel/Read/ReadVariableOpReadVariableOpdense_72/kernel*
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
VARIABLE_VALUEdense_72/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_72/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUElstm_75/lstm_cell_75/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_75/lstm_cell_75/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_75/lstm_cell_75/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_72/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_72/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_75/lstm_cell_75/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_75/lstm_cell_75/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_75/lstm_cell_75/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_72/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_72/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_75/lstm_cell_75/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_75/lstm_cell_75/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_75/lstm_cell_75/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_48Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ

Á
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_48lstm_75/lstm_cell_75/kernel%lstm_75/lstm_cell_75/recurrent_kernellstm_75/lstm_cell_75/biasdense_72/kerneldense_72/bias*
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
%__inference_signature_wrapper_3246769
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 


StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_72/kernel/Read/ReadVariableOp!dense_72/bias/Read/ReadVariableOp/lstm_75/lstm_cell_75/kernel/Read/ReadVariableOp9lstm_75/lstm_cell_75/recurrent_kernel/Read/ReadVariableOp-lstm_75/lstm_cell_75/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_72/kernel/m/Read/ReadVariableOp(Adam/dense_72/bias/m/Read/ReadVariableOp6Adam/lstm_75/lstm_cell_75/kernel/m/Read/ReadVariableOp@Adam/lstm_75/lstm_cell_75/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_75/lstm_cell_75/bias/m/Read/ReadVariableOp*Adam/dense_72/kernel/v/Read/ReadVariableOp(Adam/dense_72/bias/v/Read/ReadVariableOp6Adam/lstm_75/lstm_cell_75/kernel/v/Read/ReadVariableOp@Adam/lstm_75/lstm_cell_75/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_75/lstm_cell_75/bias/v/Read/ReadVariableOpConst*#
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
 __inference__traced_save_3247987
Ã
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_72/kerneldense_72/biaslstm_75/lstm_cell_75/kernel%lstm_75/lstm_cell_75/recurrent_kernellstm_75/lstm_cell_75/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_72/kernel/mAdam/dense_72/bias/m"Adam/lstm_75/lstm_cell_75/kernel/m,Adam/lstm_75/lstm_cell_75/recurrent_kernel/m Adam/lstm_75/lstm_cell_75/bias/mAdam/dense_72/kernel/vAdam/dense_72/bias/v"Adam/lstm_75/lstm_cell_75/kernel/v,Adam/lstm_75/lstm_cell_75/recurrent_kernel/v Adam/lstm_75/lstm_cell_75/bias/v*"
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
#__inference__traced_restore_3248063ÔÈ
Â
×
C__inference_ma_mod_layer_call_and_return_conditional_losses_3246728
input_48"
lstm_75_3246713:	#
lstm_75_3246715:

lstm_75_3246717:	#
dense_72_3246721:	

dense_72_3246723:

identity¢ dense_72/StatefulPartitionedCall¢lstm_75/StatefulPartitionedCall
lstm_75/StatefulPartitionedCallStatefulPartitionedCallinput_48lstm_75_3246713lstm_75_3246715lstm_75_3246717*
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
D__inference_lstm_75_layer_call_and_return_conditional_losses_3246385Þ
dropout_8/PartitionedCallPartitionedCall(lstm_75/StatefulPartitionedCall:output:0*
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
F__inference_dropout_8_layer_call_and_return_conditional_losses_3246398
 dense_72/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_72_3246721dense_72_3246723*
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
E__inference_dense_72_layer_call_and_return_conditional_losses_3246410ý
!weighted_layer_33/PartitionedCallPartitionedCallinput_48)dense_72/StatefulPartitionedCall:output:0*
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
N__inference_weighted_layer_33_layer_call_and_return_conditional_losses_3246424}
IdentityIdentity*weighted_layer_33/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp!^dense_72/StatefulPartitionedCall ^lstm_75/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ
: : : : : 2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2B
lstm_75/StatefulPartitionedCalllstm_75/StatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
input_48
êK

D__inference_lstm_75_layer_call_and_return_conditional_losses_3247305
inputs_0>
+lstm_cell_75_matmul_readvariableop_resource:	A
-lstm_cell_75_matmul_1_readvariableop_resource:
;
,lstm_cell_75_biasadd_readvariableop_resource:	
identity¢#lstm_cell_75/BiasAdd/ReadVariableOp¢"lstm_cell_75/MatMul/ReadVariableOp¢$lstm_cell_75/MatMul_1/ReadVariableOp¢while=
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
"lstm_cell_75/MatMul/ReadVariableOpReadVariableOp+lstm_cell_75_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_75/MatMulMatMulstrided_slice_2:output:0*lstm_cell_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_75/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_75_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_75/MatMul_1MatMulzeros:output:0,lstm_cell_75/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_75/addAddV2lstm_cell_75/MatMul:product:0lstm_cell_75/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_75/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_75_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_75/BiasAddBiasAddlstm_cell_75/add:z:0+lstm_cell_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_75/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_75/splitSplit%lstm_cell_75/split/split_dim:output:0lstm_cell_75/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_75/SigmoidSigmoidlstm_cell_75/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_75/Sigmoid_1Sigmoidlstm_cell_75/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_75/mulMullstm_cell_75/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_75/ReluRelulstm_cell_75/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_75/mul_1Mullstm_cell_75/Sigmoid:y:0lstm_cell_75/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_75/add_1AddV2lstm_cell_75/mul:z:0lstm_cell_75/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_75/Sigmoid_2Sigmoidlstm_cell_75/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_75/Relu_1Relulstm_cell_75/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_75/mul_2Mullstm_cell_75/Sigmoid_2:y:0!lstm_cell_75/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_75_matmul_readvariableop_resource-lstm_cell_75_matmul_1_readvariableop_resource,lstm_cell_75_biasadd_readvariableop_resource*
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
while_body_3247220*
condR
while_cond_3247219*M
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
NoOpNoOp$^lstm_cell_75/BiasAdd/ReadVariableOp#^lstm_cell_75/MatMul/ReadVariableOp%^lstm_cell_75/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_75/BiasAdd/ReadVariableOp#lstm_cell_75/BiasAdd/ReadVariableOp2H
"lstm_cell_75/MatMul/ReadVariableOp"lstm_cell_75/MatMul/ReadVariableOp2L
$lstm_cell_75/MatMul_1/ReadVariableOp$lstm_cell_75/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Õ
í
(__inference_ma_mod_layer_call_fn_3246784

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
C__inference_ma_mod_layer_call_and_return_conditional_losses_3246427s
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
Õ
í
(__inference_ma_mod_layer_call_fn_3246799

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
C__inference_ma_mod_layer_call_and_return_conditional_losses_3246682s
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
Ì	
÷
E__inference_dense_72_layer_call_and_return_conditional_losses_3246410

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

¹
)__inference_lstm_75_layer_call_fn_3247127
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
D__inference_lstm_75_layer_call_and_return_conditional_losses_3246032p
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
Íh

C__inference_ma_mod_layer_call_and_return_conditional_losses_3247116

inputsF
3lstm_75_lstm_cell_75_matmul_readvariableop_resource:	I
5lstm_75_lstm_cell_75_matmul_1_readvariableop_resource:
C
4lstm_75_lstm_cell_75_biasadd_readvariableop_resource:	:
'dense_72_matmul_readvariableop_resource:	
6
(dense_72_biasadd_readvariableop_resource:

identity¢dense_72/BiasAdd/ReadVariableOp¢dense_72/MatMul/ReadVariableOp¢+lstm_75/lstm_cell_75/BiasAdd/ReadVariableOp¢*lstm_75/lstm_cell_75/MatMul/ReadVariableOp¢,lstm_75/lstm_cell_75/MatMul_1/ReadVariableOp¢lstm_75/whileC
lstm_75/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_75/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_75/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_75/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_75/strided_sliceStridedSlicelstm_75/Shape:output:0$lstm_75/strided_slice/stack:output:0&lstm_75/strided_slice/stack_1:output:0&lstm_75/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_75/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_75/zeros/packedPacklstm_75/strided_slice:output:0lstm_75/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_75/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_75/zerosFilllstm_75/zeros/packed:output:0lstm_75/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm_75/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_75/zeros_1/packedPacklstm_75/strided_slice:output:0!lstm_75/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_75/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_75/zeros_1Filllstm_75/zeros_1/packed:output:0lstm_75/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_75/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_75/transpose	Transposeinputslstm_75/transpose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿT
lstm_75/Shape_1Shapelstm_75/transpose:y:0*
T0*
_output_shapes
:g
lstm_75/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_75/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_75/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_75/strided_slice_1StridedSlicelstm_75/Shape_1:output:0&lstm_75/strided_slice_1/stack:output:0(lstm_75/strided_slice_1/stack_1:output:0(lstm_75/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_75/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_75/TensorArrayV2TensorListReserve,lstm_75/TensorArrayV2/element_shape:output:0 lstm_75/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_75/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_75/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_75/transpose:y:0Flstm_75/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_75/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_75/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_75/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_75/strided_slice_2StridedSlicelstm_75/transpose:y:0&lstm_75/strided_slice_2/stack:output:0(lstm_75/strided_slice_2/stack_1:output:0(lstm_75/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
*lstm_75/lstm_cell_75/MatMul/ReadVariableOpReadVariableOp3lstm_75_lstm_cell_75_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0®
lstm_75/lstm_cell_75/MatMulMatMul lstm_75/strided_slice_2:output:02lstm_75/lstm_cell_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
,lstm_75/lstm_cell_75/MatMul_1/ReadVariableOpReadVariableOp5lstm_75_lstm_cell_75_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0¨
lstm_75/lstm_cell_75/MatMul_1MatMullstm_75/zeros:output:04lstm_75/lstm_cell_75/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
lstm_75/lstm_cell_75/addAddV2%lstm_75/lstm_cell_75/MatMul:product:0'lstm_75/lstm_cell_75/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+lstm_75/lstm_cell_75/BiasAdd/ReadVariableOpReadVariableOp4lstm_75_lstm_cell_75_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
lstm_75/lstm_cell_75/BiasAddBiasAddlstm_75/lstm_cell_75/add:z:03lstm_75/lstm_cell_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$lstm_75/lstm_cell_75/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ù
lstm_75/lstm_cell_75/splitSplit-lstm_75/lstm_cell_75/split/split_dim:output:0%lstm_75/lstm_cell_75/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
lstm_75/lstm_cell_75/SigmoidSigmoid#lstm_75/lstm_cell_75/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_75/lstm_cell_75/Sigmoid_1Sigmoid#lstm_75/lstm_cell_75/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_75/lstm_cell_75/mulMul"lstm_75/lstm_cell_75/Sigmoid_1:y:0lstm_75/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_75/lstm_cell_75/ReluRelu#lstm_75/lstm_cell_75/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_75/lstm_cell_75/mul_1Mul lstm_75/lstm_cell_75/Sigmoid:y:0'lstm_75/lstm_cell_75/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_75/lstm_cell_75/add_1AddV2lstm_75/lstm_cell_75/mul:z:0lstm_75/lstm_cell_75/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_75/lstm_cell_75/Sigmoid_2Sigmoid#lstm_75/lstm_cell_75/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
lstm_75/lstm_cell_75/Relu_1Relulstm_75/lstm_cell_75/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
lstm_75/lstm_cell_75/mul_2Mul"lstm_75/lstm_cell_75/Sigmoid_2:y:0)lstm_75/lstm_cell_75/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%lstm_75/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   f
$lstm_75/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_75/TensorArrayV2_1TensorListReserve.lstm_75/TensorArrayV2_1/element_shape:output:0-lstm_75/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_75/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_75/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_75/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ø
lstm_75/whileWhile#lstm_75/while/loop_counter:output:0)lstm_75/while/maximum_iterations:output:0lstm_75/time:output:0 lstm_75/TensorArrayV2_1:handle:0lstm_75/zeros:output:0lstm_75/zeros_1:output:0 lstm_75/strided_slice_1:output:0?lstm_75/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_75_lstm_cell_75_matmul_readvariableop_resource5lstm_75_lstm_cell_75_matmul_1_readvariableop_resource4lstm_75_lstm_cell_75_biasadd_readvariableop_resource*
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
lstm_75_while_body_3247014*&
condR
lstm_75_while_cond_3247013*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
8lstm_75/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ï
*lstm_75/TensorArrayV2Stack/TensorListStackTensorListStacklstm_75/while:output:3Alstm_75/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsp
lstm_75/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_75/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_75/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
lstm_75/strided_slice_3StridedSlice3lstm_75/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_75/strided_slice_3/stack:output:0(lstm_75/strided_slice_3/stack_1:output:0(lstm_75/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskm
lstm_75/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¯
lstm_75/transpose_1	Transpose3lstm_75/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_75/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_75/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    \
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_8/dropout/MulMul lstm_75/strided_slice_3:output:0 dropout_8/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
dropout_8/dropout/ShapeShape lstm_75/strided_slice_3:output:0*
T0*
_output_shapes
:®
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seedÒ	e
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Å
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_72/MatMul/ReadVariableOpReadVariableOp'dense_72_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
dense_72/MatMulMatMuldropout_8/dropout/Mul_1:z:0&dense_72/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_72/BiasAdd/ReadVariableOpReadVariableOp(dense_72_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_72/BiasAddBiasAdddense_72/MatMul:product:0'dense_72/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
weighted_layer_33/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
weighted_layer_33/ReshapeReshapedense_72/BiasAdd:output:0(weighted_layer_33/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
weighted_layer_33/MulMulinputs"weighted_layer_33/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
l
IdentityIdentityweighted_layer_33/Mul:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
£
NoOpNoOp ^dense_72/BiasAdd/ReadVariableOp^dense_72/MatMul/ReadVariableOp,^lstm_75/lstm_cell_75/BiasAdd/ReadVariableOp+^lstm_75/lstm_cell_75/MatMul/ReadVariableOp-^lstm_75/lstm_cell_75/MatMul_1/ReadVariableOp^lstm_75/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ
: : : : : 2B
dense_72/BiasAdd/ReadVariableOpdense_72/BiasAdd/ReadVariableOp2@
dense_72/MatMul/ReadVariableOpdense_72/MatMul/ReadVariableOp2Z
+lstm_75/lstm_cell_75/BiasAdd/ReadVariableOp+lstm_75/lstm_cell_75/BiasAdd/ReadVariableOp2X
*lstm_75/lstm_cell_75/MatMul/ReadVariableOp*lstm_75/lstm_cell_75/MatMul/ReadVariableOp2\
,lstm_75/lstm_cell_75/MatMul_1/ReadVariableOp,lstm_75/lstm_cell_75/MatMul_1/ReadVariableOp2
lstm_75/whilelstm_75/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¿9
Ó
while_body_3247510
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_75_matmul_readvariableop_resource_0:	I
5while_lstm_cell_75_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_75_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_75_matmul_readvariableop_resource:	G
3while_lstm_cell_75_matmul_1_readvariableop_resource:
A
2while_lstm_cell_75_biasadd_readvariableop_resource:	¢)while/lstm_cell_75/BiasAdd/ReadVariableOp¢(while/lstm_cell_75/MatMul/ReadVariableOp¢*while/lstm_cell_75/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_75/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_75_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_75/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_75/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_75_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_75/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_75/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/addAddV2#while/lstm_cell_75/MatMul:product:0%while/lstm_cell_75/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_75/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_75_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_75/BiasAddBiasAddwhile/lstm_cell_75/add:z:01while/lstm_cell_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_75/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_75/splitSplit+while/lstm_cell_75/split/split_dim:output:0#while/lstm_cell_75/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_75/SigmoidSigmoid!while/lstm_cell_75/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_75/Sigmoid_1Sigmoid!while/lstm_cell_75/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/mulMul while/lstm_cell_75/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_75/ReluRelu!while/lstm_cell_75/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/mul_1Mulwhile/lstm_cell_75/Sigmoid:y:0%while/lstm_cell_75/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/add_1AddV2while/lstm_cell_75/mul:z:0while/lstm_cell_75/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_75/Sigmoid_2Sigmoid!while/lstm_cell_75/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_75/Relu_1Reluwhile/lstm_cell_75/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/mul_2Mul while/lstm_cell_75/Sigmoid_2:y:0'while/lstm_cell_75/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_75/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_75/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_75/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_75/BiasAdd/ReadVariableOp)^while/lstm_cell_75/MatMul/ReadVariableOp+^while/lstm_cell_75/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_75_biasadd_readvariableop_resource4while_lstm_cell_75_biasadd_readvariableop_resource_0"l
3while_lstm_cell_75_matmul_1_readvariableop_resource5while_lstm_cell_75_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_75_matmul_readvariableop_resource3while_lstm_cell_75_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_75/BiasAdd/ReadVariableOp)while/lstm_cell_75/BiasAdd/ReadVariableOp2T
(while/lstm_cell_75/MatMul/ReadVariableOp(while/lstm_cell_75/MatMul/ReadVariableOp2X
*while/lstm_cell_75/MatMul_1/ReadVariableOp*while/lstm_cell_75/MatMul_1/ReadVariableOp: 
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
â
ù
C__inference_ma_mod_layer_call_and_return_conditional_losses_3246682

inputs"
lstm_75_3246667:	#
lstm_75_3246669:

lstm_75_3246671:	#
dense_72_3246675:	

dense_72_3246677:

identity¢ dense_72/StatefulPartitionedCall¢!dropout_8/StatefulPartitionedCall¢lstm_75/StatefulPartitionedCall
lstm_75/StatefulPartitionedCallStatefulPartitionedCallinputslstm_75_3246667lstm_75_3246669lstm_75_3246671*
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
D__inference_lstm_75_layer_call_and_return_conditional_losses_3246638î
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(lstm_75/StatefulPartitionedCall:output:0*
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
F__inference_dropout_8_layer_call_and_return_conditional_losses_3246477
 dense_72/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_72_3246675dense_72_3246677*
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
E__inference_dense_72_layer_call_and_return_conditional_losses_3246410û
!weighted_layer_33/PartitionedCallPartitionedCallinputs)dense_72/StatefulPartitionedCall:output:0*
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
N__inference_weighted_layer_33_layer_call_and_return_conditional_losses_3246424}
IdentityIdentity*weighted_layer_33/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¯
NoOpNoOp!^dense_72/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall ^lstm_75/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ
: : : : : 2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2B
lstm_75/StatefulPartitionedCalllstm_75/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
·
ì
%__inference_signature_wrapper_3246769
input_48
unknown:	
	unknown_0:

	unknown_1:	
	unknown_2:	

	unknown_3:

identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinput_48unknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
"__inference__wrapped_model_3245880s
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
input_48
¾
È
while_cond_3245961
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3245961___redundant_placeholder05
1while_while_cond_3245961___redundant_placeholder15
1while_while_cond_3245961___redundant_placeholder25
1while_while_cond_3245961___redundant_placeholder3
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
while_cond_3247509
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3247509___redundant_placeholder05
1while_while_cond_3247509___redundant_placeholder15
1while_while_cond_3247509___redundant_placeholder25
1while_while_cond_3247509___redundant_placeholder3
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
ì

I__inference_lstm_cell_75_layer_call_and_return_conditional_losses_3245947

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
9

D__inference_lstm_75_layer_call_and_return_conditional_losses_3246225

inputs'
lstm_cell_75_3246141:	(
lstm_cell_75_3246143:
#
lstm_cell_75_3246145:	
identity¢$lstm_cell_75/StatefulPartitionedCall¢while;
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
$lstm_cell_75/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_75_3246141lstm_cell_75_3246143lstm_cell_75_3246145*
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
I__inference_lstm_cell_75_layer_call_and_return_conditional_losses_3246095n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_75_3246141lstm_cell_75_3246143lstm_cell_75_3246145*
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
while_body_3246155*
condR
while_cond_3246154*M
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
NoOpNoOp%^lstm_cell_75/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_75/StatefulPartitionedCall$lstm_cell_75/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô

I__inference_lstm_cell_75_layer_call_and_return_conditional_losses_3247898

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
D__inference_lstm_75_layer_call_and_return_conditional_losses_3246032

inputs'
lstm_cell_75_3245948:	(
lstm_cell_75_3245950:
#
lstm_cell_75_3245952:	
identity¢$lstm_cell_75/StatefulPartitionedCall¢while;
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
$lstm_cell_75/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_75_3245948lstm_cell_75_3245950lstm_cell_75_3245952*
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
I__inference_lstm_cell_75_layer_call_and_return_conditional_losses_3245947n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_75_3245948lstm_cell_75_3245950lstm_cell_75_3245952*
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
while_body_3245962*
condR
while_cond_3245961*M
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
NoOpNoOp%^lstm_cell_75/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_75/StatefulPartitionedCall$lstm_cell_75/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
·
)__inference_lstm_75_layer_call_fn_3247149

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
D__inference_lstm_75_layer_call_and_return_conditional_losses_3246385p
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


è
lstm_75_while_cond_3246858,
(lstm_75_while_lstm_75_while_loop_counter2
.lstm_75_while_lstm_75_while_maximum_iterations
lstm_75_while_placeholder
lstm_75_while_placeholder_1
lstm_75_while_placeholder_2
lstm_75_while_placeholder_3.
*lstm_75_while_less_lstm_75_strided_slice_1E
Alstm_75_while_lstm_75_while_cond_3246858___redundant_placeholder0E
Alstm_75_while_lstm_75_while_cond_3246858___redundant_placeholder1E
Alstm_75_while_lstm_75_while_cond_3246858___redundant_placeholder2E
Alstm_75_while_lstm_75_while_cond_3246858___redundant_placeholder3
lstm_75_while_identity

lstm_75/while/LessLesslstm_75_while_placeholder*lstm_75_while_less_lstm_75_strided_slice_1*
T0*
_output_shapes
: [
lstm_75/while/IdentityIdentitylstm_75/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_75_while_identitylstm_75/while/Identity:output:0*(
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
(__inference_ma_mod_layer_call_fn_3246440
input_48
unknown:	
	unknown_0:

	unknown_1:	
	unknown_2:	

	unknown_3:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_48unknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
C__inference_ma_mod_layer_call_and_return_conditional_losses_3246427s
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
input_48
Ç

*__inference_dense_72_layer_call_fn_3247776

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
E__inference_dense_72_layer_call_and_return_conditional_losses_3246410o
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
º
Õ
C__inference_ma_mod_layer_call_and_return_conditional_losses_3246427

inputs"
lstm_75_3246386:	#
lstm_75_3246388:

lstm_75_3246390:	#
dense_72_3246411:	

dense_72_3246413:

identity¢ dense_72/StatefulPartitionedCall¢lstm_75/StatefulPartitionedCall
lstm_75/StatefulPartitionedCallStatefulPartitionedCallinputslstm_75_3246386lstm_75_3246388lstm_75_3246390*
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
D__inference_lstm_75_layer_call_and_return_conditional_losses_3246385Þ
dropout_8/PartitionedCallPartitionedCall(lstm_75/StatefulPartitionedCall:output:0*
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
F__inference_dropout_8_layer_call_and_return_conditional_losses_3246398
 dense_72/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_72_3246411dense_72_3246413*
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
E__inference_dense_72_layer_call_and_return_conditional_losses_3246410û
!weighted_layer_33/PartitionedCallPartitionedCallinputs)dense_72/StatefulPartitionedCall:output:0*
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
N__inference_weighted_layer_33_layer_call_and_return_conditional_losses_3246424}
IdentityIdentity*weighted_layer_33/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp!^dense_72/StatefulPartitionedCall ^lstm_75/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ
: : : : : 2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2B
lstm_75/StatefulPartitionedCalllstm_75/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ÇK

D__inference_lstm_75_layer_call_and_return_conditional_losses_3247595

inputs>
+lstm_cell_75_matmul_readvariableop_resource:	A
-lstm_cell_75_matmul_1_readvariableop_resource:
;
,lstm_cell_75_biasadd_readvariableop_resource:	
identity¢#lstm_cell_75/BiasAdd/ReadVariableOp¢"lstm_cell_75/MatMul/ReadVariableOp¢$lstm_cell_75/MatMul_1/ReadVariableOp¢while;
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
"lstm_cell_75/MatMul/ReadVariableOpReadVariableOp+lstm_cell_75_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_75/MatMulMatMulstrided_slice_2:output:0*lstm_cell_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_75/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_75_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_75/MatMul_1MatMulzeros:output:0,lstm_cell_75/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_75/addAddV2lstm_cell_75/MatMul:product:0lstm_cell_75/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_75/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_75_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_75/BiasAddBiasAddlstm_cell_75/add:z:0+lstm_cell_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_75/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_75/splitSplit%lstm_cell_75/split/split_dim:output:0lstm_cell_75/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_75/SigmoidSigmoidlstm_cell_75/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_75/Sigmoid_1Sigmoidlstm_cell_75/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_75/mulMullstm_cell_75/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_75/ReluRelulstm_cell_75/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_75/mul_1Mullstm_cell_75/Sigmoid:y:0lstm_cell_75/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_75/add_1AddV2lstm_cell_75/mul:z:0lstm_cell_75/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_75/Sigmoid_2Sigmoidlstm_cell_75/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_75/Relu_1Relulstm_cell_75/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_75/mul_2Mullstm_cell_75/Sigmoid_2:y:0!lstm_cell_75/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_75_matmul_readvariableop_resource-lstm_cell_75_matmul_1_readvariableop_resource,lstm_cell_75_biasadd_readvariableop_resource*
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
while_body_3247510*
condR
while_cond_3247509*M
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
NoOpNoOp$^lstm_cell_75/BiasAdd/ReadVariableOp#^lstm_cell_75/MatMul/ReadVariableOp%^lstm_cell_75/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 2J
#lstm_cell_75/BiasAdd/ReadVariableOp#lstm_cell_75/BiasAdd/ReadVariableOp2H
"lstm_cell_75/MatMul/ReadVariableOp"lstm_cell_75/MatMul/ReadVariableOp2L
$lstm_cell_75/MatMul_1/ReadVariableOp$lstm_cell_75/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ø
ø
.__inference_lstm_cell_75_layer_call_fn_3247817

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
I__inference_lstm_cell_75_layer_call_and_return_conditional_losses_3245947p
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
¾
È
while_cond_3247364
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3247364___redundant_placeholder05
1while_while_cond_3247364___redundant_placeholder15
1while_while_cond_3247364___redundant_placeholder25
1while_while_cond_3247364___redundant_placeholder3
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
while_body_3247655
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_75_matmul_readvariableop_resource_0:	I
5while_lstm_cell_75_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_75_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_75_matmul_readvariableop_resource:	G
3while_lstm_cell_75_matmul_1_readvariableop_resource:
A
2while_lstm_cell_75_biasadd_readvariableop_resource:	¢)while/lstm_cell_75/BiasAdd/ReadVariableOp¢(while/lstm_cell_75/MatMul/ReadVariableOp¢*while/lstm_cell_75/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_75/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_75_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_75/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_75/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_75_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_75/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_75/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/addAddV2#while/lstm_cell_75/MatMul:product:0%while/lstm_cell_75/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_75/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_75_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_75/BiasAddBiasAddwhile/lstm_cell_75/add:z:01while/lstm_cell_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_75/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_75/splitSplit+while/lstm_cell_75/split/split_dim:output:0#while/lstm_cell_75/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_75/SigmoidSigmoid!while/lstm_cell_75/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_75/Sigmoid_1Sigmoid!while/lstm_cell_75/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/mulMul while/lstm_cell_75/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_75/ReluRelu!while/lstm_cell_75/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/mul_1Mulwhile/lstm_cell_75/Sigmoid:y:0%while/lstm_cell_75/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/add_1AddV2while/lstm_cell_75/mul:z:0while/lstm_cell_75/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_75/Sigmoid_2Sigmoid!while/lstm_cell_75/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_75/Relu_1Reluwhile/lstm_cell_75/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/mul_2Mul while/lstm_cell_75/Sigmoid_2:y:0'while/lstm_cell_75/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_75/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_75/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_75/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_75/BiasAdd/ReadVariableOp)^while/lstm_cell_75/MatMul/ReadVariableOp+^while/lstm_cell_75/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_75_biasadd_readvariableop_resource4while_lstm_cell_75_biasadd_readvariableop_resource_0"l
3while_lstm_cell_75_matmul_1_readvariableop_resource5while_lstm_cell_75_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_75_matmul_readvariableop_resource3while_lstm_cell_75_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_75/BiasAdd/ReadVariableOp)while/lstm_cell_75/BiasAdd/ReadVariableOp2T
(while/lstm_cell_75/MatMul/ReadVariableOp(while/lstm_cell_75/MatMul/ReadVariableOp2X
*while/lstm_cell_75/MatMul_1/ReadVariableOp*while/lstm_cell_75/MatMul_1/ReadVariableOp: 
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
¾
È
while_cond_3246552
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3246552___redundant_placeholder05
1while_while_cond_3246552___redundant_placeholder15
1while_while_cond_3246552___redundant_placeholder25
1while_while_cond_3246552___redundant_placeholder3
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
ã
ô
!ma_mod_lstm_75_while_cond_3245784:
6ma_mod_lstm_75_while_ma_mod_lstm_75_while_loop_counter@
<ma_mod_lstm_75_while_ma_mod_lstm_75_while_maximum_iterations$
 ma_mod_lstm_75_while_placeholder&
"ma_mod_lstm_75_while_placeholder_1&
"ma_mod_lstm_75_while_placeholder_2&
"ma_mod_lstm_75_while_placeholder_3<
8ma_mod_lstm_75_while_less_ma_mod_lstm_75_strided_slice_1S
Oma_mod_lstm_75_while_ma_mod_lstm_75_while_cond_3245784___redundant_placeholder0S
Oma_mod_lstm_75_while_ma_mod_lstm_75_while_cond_3245784___redundant_placeholder1S
Oma_mod_lstm_75_while_ma_mod_lstm_75_while_cond_3245784___redundant_placeholder2S
Oma_mod_lstm_75_while_ma_mod_lstm_75_while_cond_3245784___redundant_placeholder3!
ma_mod_lstm_75_while_identity

ma_mod/lstm_75/while/LessLess ma_mod_lstm_75_while_placeholder8ma_mod_lstm_75_while_less_ma_mod_lstm_75_strided_slice_1*
T0*
_output_shapes
: i
ma_mod/lstm_75/while/IdentityIdentityma_mod/lstm_75/while/Less:z:0*
T0
*
_output_shapes
: "G
ma_mod_lstm_75_while_identity&ma_mod/lstm_75/while/Identity:output:0*(
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
ê
û
C__inference_ma_mod_layer_call_and_return_conditional_losses_3246746
input_48"
lstm_75_3246731:	#
lstm_75_3246733:

lstm_75_3246735:	#
dense_72_3246739:	

dense_72_3246741:

identity¢ dense_72/StatefulPartitionedCall¢!dropout_8/StatefulPartitionedCall¢lstm_75/StatefulPartitionedCall
lstm_75/StatefulPartitionedCallStatefulPartitionedCallinput_48lstm_75_3246731lstm_75_3246733lstm_75_3246735*
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
D__inference_lstm_75_layer_call_and_return_conditional_losses_3246638î
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(lstm_75/StatefulPartitionedCall:output:0*
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
F__inference_dropout_8_layer_call_and_return_conditional_losses_3246477
 dense_72/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_72_3246739dense_72_3246741*
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
E__inference_dense_72_layer_call_and_return_conditional_losses_3246410ý
!weighted_layer_33/PartitionedCallPartitionedCallinput_48)dense_72/StatefulPartitionedCall:output:0*
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
N__inference_weighted_layer_33_layer_call_and_return_conditional_losses_3246424}
IdentityIdentity*weighted_layer_33/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¯
NoOpNoOp!^dense_72/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall ^lstm_75/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ
: : : : : 2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2B
lstm_75/StatefulPartitionedCalllstm_75/StatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
input_48
¿9
Ó
while_body_3247365
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_75_matmul_readvariableop_resource_0:	I
5while_lstm_cell_75_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_75_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_75_matmul_readvariableop_resource:	G
3while_lstm_cell_75_matmul_1_readvariableop_resource:
A
2while_lstm_cell_75_biasadd_readvariableop_resource:	¢)while/lstm_cell_75/BiasAdd/ReadVariableOp¢(while/lstm_cell_75/MatMul/ReadVariableOp¢*while/lstm_cell_75/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_75/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_75_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_75/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_75/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_75_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_75/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_75/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/addAddV2#while/lstm_cell_75/MatMul:product:0%while/lstm_cell_75/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_75/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_75_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_75/BiasAddBiasAddwhile/lstm_cell_75/add:z:01while/lstm_cell_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_75/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_75/splitSplit+while/lstm_cell_75/split/split_dim:output:0#while/lstm_cell_75/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_75/SigmoidSigmoid!while/lstm_cell_75/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_75/Sigmoid_1Sigmoid!while/lstm_cell_75/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/mulMul while/lstm_cell_75/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_75/ReluRelu!while/lstm_cell_75/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/mul_1Mulwhile/lstm_cell_75/Sigmoid:y:0%while/lstm_cell_75/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/add_1AddV2while/lstm_cell_75/mul:z:0while/lstm_cell_75/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_75/Sigmoid_2Sigmoid!while/lstm_cell_75/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_75/Relu_1Reluwhile/lstm_cell_75/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/mul_2Mul while/lstm_cell_75/Sigmoid_2:y:0'while/lstm_cell_75/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_75/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_75/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_75/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_75/BiasAdd/ReadVariableOp)^while/lstm_cell_75/MatMul/ReadVariableOp+^while/lstm_cell_75/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_75_biasadd_readvariableop_resource4while_lstm_cell_75_biasadd_readvariableop_resource_0"l
3while_lstm_cell_75_matmul_1_readvariableop_resource5while_lstm_cell_75_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_75_matmul_readvariableop_resource3while_lstm_cell_75_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_75/BiasAdd/ReadVariableOp)while/lstm_cell_75/BiasAdd/ReadVariableOp2T
(while/lstm_cell_75/MatMul/ReadVariableOp(while/lstm_cell_75/MatMul/ReadVariableOp2X
*while/lstm_cell_75/MatMul_1/ReadVariableOp*while/lstm_cell_75/MatMul_1/ReadVariableOp: 
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
Ì	
÷
E__inference_dense_72_layer_call_and_return_conditional_losses_3247786

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
¡
x
N__inference_weighted_layer_33_layer_call_and_return_conditional_losses_3246424

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
¾
È
while_cond_3247219
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3247219___redundant_placeholder05
1while_while_cond_3247219___redundant_placeholder15
1while_while_cond_3247219___redundant_placeholder25
1while_while_cond_3247219___redundant_placeholder3
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
ÇK

D__inference_lstm_75_layer_call_and_return_conditional_losses_3246638

inputs>
+lstm_cell_75_matmul_readvariableop_resource:	A
-lstm_cell_75_matmul_1_readvariableop_resource:
;
,lstm_cell_75_biasadd_readvariableop_resource:	
identity¢#lstm_cell_75/BiasAdd/ReadVariableOp¢"lstm_cell_75/MatMul/ReadVariableOp¢$lstm_cell_75/MatMul_1/ReadVariableOp¢while;
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
"lstm_cell_75/MatMul/ReadVariableOpReadVariableOp+lstm_cell_75_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_75/MatMulMatMulstrided_slice_2:output:0*lstm_cell_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_75/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_75_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_75/MatMul_1MatMulzeros:output:0,lstm_cell_75/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_75/addAddV2lstm_cell_75/MatMul:product:0lstm_cell_75/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_75/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_75_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_75/BiasAddBiasAddlstm_cell_75/add:z:0+lstm_cell_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_75/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_75/splitSplit%lstm_cell_75/split/split_dim:output:0lstm_cell_75/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_75/SigmoidSigmoidlstm_cell_75/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_75/Sigmoid_1Sigmoidlstm_cell_75/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_75/mulMullstm_cell_75/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_75/ReluRelulstm_cell_75/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_75/mul_1Mullstm_cell_75/Sigmoid:y:0lstm_cell_75/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_75/add_1AddV2lstm_cell_75/mul:z:0lstm_cell_75/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_75/Sigmoid_2Sigmoidlstm_cell_75/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_75/Relu_1Relulstm_cell_75/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_75/mul_2Mullstm_cell_75/Sigmoid_2:y:0!lstm_cell_75/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_75_matmul_readvariableop_resource-lstm_cell_75_matmul_1_readvariableop_resource,lstm_cell_75_biasadd_readvariableop_resource*
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
while_body_3246553*
condR
while_cond_3246552*M
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
NoOpNoOp$^lstm_cell_75/BiasAdd/ReadVariableOp#^lstm_cell_75/MatMul/ReadVariableOp%^lstm_cell_75/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 2J
#lstm_cell_75/BiasAdd/ReadVariableOp#lstm_cell_75/BiasAdd/ReadVariableOp2H
"lstm_cell_75/MatMul/ReadVariableOp"lstm_cell_75/MatMul/ReadVariableOp2L
$lstm_cell_75/MatMul_1/ReadVariableOp$lstm_cell_75/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¿9
Ó
while_body_3247220
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_75_matmul_readvariableop_resource_0:	I
5while_lstm_cell_75_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_75_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_75_matmul_readvariableop_resource:	G
3while_lstm_cell_75_matmul_1_readvariableop_resource:
A
2while_lstm_cell_75_biasadd_readvariableop_resource:	¢)while/lstm_cell_75/BiasAdd/ReadVariableOp¢(while/lstm_cell_75/MatMul/ReadVariableOp¢*while/lstm_cell_75/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_75/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_75_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_75/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_75/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_75_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_75/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_75/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/addAddV2#while/lstm_cell_75/MatMul:product:0%while/lstm_cell_75/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_75/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_75_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_75/BiasAddBiasAddwhile/lstm_cell_75/add:z:01while/lstm_cell_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_75/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_75/splitSplit+while/lstm_cell_75/split/split_dim:output:0#while/lstm_cell_75/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_75/SigmoidSigmoid!while/lstm_cell_75/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_75/Sigmoid_1Sigmoid!while/lstm_cell_75/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/mulMul while/lstm_cell_75/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_75/ReluRelu!while/lstm_cell_75/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/mul_1Mulwhile/lstm_cell_75/Sigmoid:y:0%while/lstm_cell_75/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/add_1AddV2while/lstm_cell_75/mul:z:0while/lstm_cell_75/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_75/Sigmoid_2Sigmoid!while/lstm_cell_75/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_75/Relu_1Reluwhile/lstm_cell_75/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/mul_2Mul while/lstm_cell_75/Sigmoid_2:y:0'while/lstm_cell_75/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_75/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_75/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_75/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_75/BiasAdd/ReadVariableOp)^while/lstm_cell_75/MatMul/ReadVariableOp+^while/lstm_cell_75/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_75_biasadd_readvariableop_resource4while_lstm_cell_75_biasadd_readvariableop_resource_0"l
3while_lstm_cell_75_matmul_1_readvariableop_resource5while_lstm_cell_75_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_75_matmul_readvariableop_resource3while_lstm_cell_75_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_75/BiasAdd/ReadVariableOp)while/lstm_cell_75/BiasAdd/ReadVariableOp2T
(while/lstm_cell_75/MatMul/ReadVariableOp(while/lstm_cell_75/MatMul/ReadVariableOp2X
*while/lstm_cell_75/MatMul_1/ReadVariableOp*while/lstm_cell_75/MatMul_1/ReadVariableOp: 
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
©
z
N__inference_weighted_layer_33_layer_call_and_return_conditional_losses_3247800
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
ó`

C__inference_ma_mod_layer_call_and_return_conditional_losses_3246954

inputsF
3lstm_75_lstm_cell_75_matmul_readvariableop_resource:	I
5lstm_75_lstm_cell_75_matmul_1_readvariableop_resource:
C
4lstm_75_lstm_cell_75_biasadd_readvariableop_resource:	:
'dense_72_matmul_readvariableop_resource:	
6
(dense_72_biasadd_readvariableop_resource:

identity¢dense_72/BiasAdd/ReadVariableOp¢dense_72/MatMul/ReadVariableOp¢+lstm_75/lstm_cell_75/BiasAdd/ReadVariableOp¢*lstm_75/lstm_cell_75/MatMul/ReadVariableOp¢,lstm_75/lstm_cell_75/MatMul_1/ReadVariableOp¢lstm_75/whileC
lstm_75/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_75/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_75/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_75/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_75/strided_sliceStridedSlicelstm_75/Shape:output:0$lstm_75/strided_slice/stack:output:0&lstm_75/strided_slice/stack_1:output:0&lstm_75/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_75/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_75/zeros/packedPacklstm_75/strided_slice:output:0lstm_75/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_75/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_75/zerosFilllstm_75/zeros/packed:output:0lstm_75/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm_75/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_75/zeros_1/packedPacklstm_75/strided_slice:output:0!lstm_75/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_75/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_75/zeros_1Filllstm_75/zeros_1/packed:output:0lstm_75/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_75/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_75/transpose	Transposeinputslstm_75/transpose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿT
lstm_75/Shape_1Shapelstm_75/transpose:y:0*
T0*
_output_shapes
:g
lstm_75/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_75/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_75/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_75/strided_slice_1StridedSlicelstm_75/Shape_1:output:0&lstm_75/strided_slice_1/stack:output:0(lstm_75/strided_slice_1/stack_1:output:0(lstm_75/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_75/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_75/TensorArrayV2TensorListReserve,lstm_75/TensorArrayV2/element_shape:output:0 lstm_75/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_75/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_75/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_75/transpose:y:0Flstm_75/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_75/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_75/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_75/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_75/strided_slice_2StridedSlicelstm_75/transpose:y:0&lstm_75/strided_slice_2/stack:output:0(lstm_75/strided_slice_2/stack_1:output:0(lstm_75/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
*lstm_75/lstm_cell_75/MatMul/ReadVariableOpReadVariableOp3lstm_75_lstm_cell_75_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0®
lstm_75/lstm_cell_75/MatMulMatMul lstm_75/strided_slice_2:output:02lstm_75/lstm_cell_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
,lstm_75/lstm_cell_75/MatMul_1/ReadVariableOpReadVariableOp5lstm_75_lstm_cell_75_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0¨
lstm_75/lstm_cell_75/MatMul_1MatMullstm_75/zeros:output:04lstm_75/lstm_cell_75/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
lstm_75/lstm_cell_75/addAddV2%lstm_75/lstm_cell_75/MatMul:product:0'lstm_75/lstm_cell_75/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+lstm_75/lstm_cell_75/BiasAdd/ReadVariableOpReadVariableOp4lstm_75_lstm_cell_75_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
lstm_75/lstm_cell_75/BiasAddBiasAddlstm_75/lstm_cell_75/add:z:03lstm_75/lstm_cell_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$lstm_75/lstm_cell_75/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ù
lstm_75/lstm_cell_75/splitSplit-lstm_75/lstm_cell_75/split/split_dim:output:0%lstm_75/lstm_cell_75/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
lstm_75/lstm_cell_75/SigmoidSigmoid#lstm_75/lstm_cell_75/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_75/lstm_cell_75/Sigmoid_1Sigmoid#lstm_75/lstm_cell_75/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_75/lstm_cell_75/mulMul"lstm_75/lstm_cell_75/Sigmoid_1:y:0lstm_75/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_75/lstm_cell_75/ReluRelu#lstm_75/lstm_cell_75/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_75/lstm_cell_75/mul_1Mul lstm_75/lstm_cell_75/Sigmoid:y:0'lstm_75/lstm_cell_75/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_75/lstm_cell_75/add_1AddV2lstm_75/lstm_cell_75/mul:z:0lstm_75/lstm_cell_75/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_75/lstm_cell_75/Sigmoid_2Sigmoid#lstm_75/lstm_cell_75/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
lstm_75/lstm_cell_75/Relu_1Relulstm_75/lstm_cell_75/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
lstm_75/lstm_cell_75/mul_2Mul"lstm_75/lstm_cell_75/Sigmoid_2:y:0)lstm_75/lstm_cell_75/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%lstm_75/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   f
$lstm_75/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_75/TensorArrayV2_1TensorListReserve.lstm_75/TensorArrayV2_1/element_shape:output:0-lstm_75/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_75/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_75/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_75/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ø
lstm_75/whileWhile#lstm_75/while/loop_counter:output:0)lstm_75/while/maximum_iterations:output:0lstm_75/time:output:0 lstm_75/TensorArrayV2_1:handle:0lstm_75/zeros:output:0lstm_75/zeros_1:output:0 lstm_75/strided_slice_1:output:0?lstm_75/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_75_lstm_cell_75_matmul_readvariableop_resource5lstm_75_lstm_cell_75_matmul_1_readvariableop_resource4lstm_75_lstm_cell_75_biasadd_readvariableop_resource*
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
lstm_75_while_body_3246859*&
condR
lstm_75_while_cond_3246858*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
8lstm_75/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ï
*lstm_75/TensorArrayV2Stack/TensorListStackTensorListStacklstm_75/while:output:3Alstm_75/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsp
lstm_75/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_75/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_75/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
lstm_75/strided_slice_3StridedSlice3lstm_75/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_75/strided_slice_3/stack:output:0(lstm_75/strided_slice_3/stack_1:output:0(lstm_75/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskm
lstm_75/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¯
lstm_75/transpose_1	Transpose3lstm_75/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_75/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_75/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    s
dropout_8/IdentityIdentity lstm_75/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_72/MatMul/ReadVariableOpReadVariableOp'dense_72_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
dense_72/MatMulMatMuldropout_8/Identity:output:0&dense_72/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_72/BiasAdd/ReadVariableOpReadVariableOp(dense_72_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_72/BiasAddBiasAdddense_72/MatMul:product:0'dense_72/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
weighted_layer_33/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
weighted_layer_33/ReshapeReshapedense_72/BiasAdd:output:0(weighted_layer_33/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
weighted_layer_33/MulMulinputs"weighted_layer_33/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
l
IdentityIdentityweighted_layer_33/Mul:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
£
NoOpNoOp ^dense_72/BiasAdd/ReadVariableOp^dense_72/MatMul/ReadVariableOp,^lstm_75/lstm_cell_75/BiasAdd/ReadVariableOp+^lstm_75/lstm_cell_75/MatMul/ReadVariableOp-^lstm_75/lstm_cell_75/MatMul_1/ReadVariableOp^lstm_75/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ
: : : : : 2B
dense_72/BiasAdd/ReadVariableOpdense_72/BiasAdd/ReadVariableOp2@
dense_72/MatMul/ReadVariableOpdense_72/MatMul/ReadVariableOp2Z
+lstm_75/lstm_cell_75/BiasAdd/ReadVariableOp+lstm_75/lstm_cell_75/BiasAdd/ReadVariableOp2X
*lstm_75/lstm_cell_75/MatMul/ReadVariableOp*lstm_75/lstm_cell_75/MatMul/ReadVariableOp2\
,lstm_75/lstm_cell_75/MatMul_1/ReadVariableOp,lstm_75/lstm_cell_75/MatMul_1/ReadVariableOp2
lstm_75/whilelstm_75/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
º5
­

 __inference__traced_save_3247987
file_prefix.
*savev2_dense_72_kernel_read_readvariableop,
(savev2_dense_72_bias_read_readvariableop:
6savev2_lstm_75_lstm_cell_75_kernel_read_readvariableopD
@savev2_lstm_75_lstm_cell_75_recurrent_kernel_read_readvariableop8
4savev2_lstm_75_lstm_cell_75_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_72_kernel_m_read_readvariableop3
/savev2_adam_dense_72_bias_m_read_readvariableopA
=savev2_adam_lstm_75_lstm_cell_75_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_75_lstm_cell_75_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_75_lstm_cell_75_bias_m_read_readvariableop5
1savev2_adam_dense_72_kernel_v_read_readvariableop3
/savev2_adam_dense_72_bias_v_read_readvariableopA
=savev2_adam_lstm_75_lstm_cell_75_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_75_lstm_cell_75_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_75_lstm_cell_75_bias_v_read_readvariableop
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

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_72_kernel_read_readvariableop(savev2_dense_72_bias_read_readvariableop6savev2_lstm_75_lstm_cell_75_kernel_read_readvariableop@savev2_lstm_75_lstm_cell_75_recurrent_kernel_read_readvariableop4savev2_lstm_75_lstm_cell_75_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_72_kernel_m_read_readvariableop/savev2_adam_dense_72_bias_m_read_readvariableop=savev2_adam_lstm_75_lstm_cell_75_kernel_m_read_readvariableopGsavev2_adam_lstm_75_lstm_cell_75_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_75_lstm_cell_75_bias_m_read_readvariableop1savev2_adam_dense_72_kernel_v_read_readvariableop/savev2_adam_dense_72_bias_v_read_readvariableop=savev2_adam_lstm_75_lstm_cell_75_kernel_v_read_readvariableopGsavev2_adam_lstm_75_lstm_cell_75_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_75_lstm_cell_75_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
C
Ó

lstm_75_while_body_3247014,
(lstm_75_while_lstm_75_while_loop_counter2
.lstm_75_while_lstm_75_while_maximum_iterations
lstm_75_while_placeholder
lstm_75_while_placeholder_1
lstm_75_while_placeholder_2
lstm_75_while_placeholder_3+
'lstm_75_while_lstm_75_strided_slice_1_0g
clstm_75_while_tensorarrayv2read_tensorlistgetitem_lstm_75_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_75_while_lstm_cell_75_matmul_readvariableop_resource_0:	Q
=lstm_75_while_lstm_cell_75_matmul_1_readvariableop_resource_0:
K
<lstm_75_while_lstm_cell_75_biasadd_readvariableop_resource_0:	
lstm_75_while_identity
lstm_75_while_identity_1
lstm_75_while_identity_2
lstm_75_while_identity_3
lstm_75_while_identity_4
lstm_75_while_identity_5)
%lstm_75_while_lstm_75_strided_slice_1e
alstm_75_while_tensorarrayv2read_tensorlistgetitem_lstm_75_tensorarrayunstack_tensorlistfromtensorL
9lstm_75_while_lstm_cell_75_matmul_readvariableop_resource:	O
;lstm_75_while_lstm_cell_75_matmul_1_readvariableop_resource:
I
:lstm_75_while_lstm_cell_75_biasadd_readvariableop_resource:	¢1lstm_75/while/lstm_cell_75/BiasAdd/ReadVariableOp¢0lstm_75/while/lstm_cell_75/MatMul/ReadVariableOp¢2lstm_75/while/lstm_cell_75/MatMul_1/ReadVariableOp
?lstm_75/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Î
1lstm_75/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_75_while_tensorarrayv2read_tensorlistgetitem_lstm_75_tensorarrayunstack_tensorlistfromtensor_0lstm_75_while_placeholderHlstm_75/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0­
0lstm_75/while/lstm_cell_75/MatMul/ReadVariableOpReadVariableOp;lstm_75_while_lstm_cell_75_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ò
!lstm_75/while/lstm_cell_75/MatMulMatMul8lstm_75/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_75/while/lstm_cell_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
2lstm_75/while/lstm_cell_75/MatMul_1/ReadVariableOpReadVariableOp=lstm_75_while_lstm_cell_75_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¹
#lstm_75/while/lstm_cell_75/MatMul_1MatMullstm_75_while_placeholder_2:lstm_75/while/lstm_cell_75/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm_75/while/lstm_cell_75/addAddV2+lstm_75/while/lstm_cell_75/MatMul:product:0-lstm_75/while/lstm_cell_75/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
1lstm_75/while/lstm_cell_75/BiasAdd/ReadVariableOpReadVariableOp<lstm_75_while_lstm_cell_75_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¿
"lstm_75/while/lstm_cell_75/BiasAddBiasAdd"lstm_75/while/lstm_cell_75/add:z:09lstm_75/while/lstm_cell_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*lstm_75/while/lstm_cell_75/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_75/while/lstm_cell_75/splitSplit3lstm_75/while/lstm_cell_75/split/split_dim:output:0+lstm_75/while/lstm_cell_75/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"lstm_75/while/lstm_cell_75/SigmoidSigmoid)lstm_75/while/lstm_cell_75/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_75/while/lstm_cell_75/Sigmoid_1Sigmoid)lstm_75/while/lstm_cell_75/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_75/while/lstm_cell_75/mulMul(lstm_75/while/lstm_cell_75/Sigmoid_1:y:0lstm_75_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_75/while/lstm_cell_75/ReluRelu)lstm_75/while/lstm_cell_75/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
 lstm_75/while/lstm_cell_75/mul_1Mul&lstm_75/while/lstm_cell_75/Sigmoid:y:0-lstm_75/while/lstm_cell_75/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
 lstm_75/while/lstm_cell_75/add_1AddV2"lstm_75/while/lstm_cell_75/mul:z:0$lstm_75/while/lstm_cell_75/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_75/while/lstm_cell_75/Sigmoid_2Sigmoid)lstm_75/while/lstm_cell_75/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_75/while/lstm_cell_75/Relu_1Relu$lstm_75/while/lstm_cell_75/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
 lstm_75/while/lstm_cell_75/mul_2Mul(lstm_75/while/lstm_cell_75/Sigmoid_2:y:0/lstm_75/while/lstm_cell_75/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8lstm_75/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
2lstm_75/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_75_while_placeholder_1Alstm_75/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_75/while/lstm_cell_75/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_75/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_75/while/addAddV2lstm_75_while_placeholderlstm_75/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_75/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_75/while/add_1AddV2(lstm_75_while_lstm_75_while_loop_counterlstm_75/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_75/while/IdentityIdentitylstm_75/while/add_1:z:0^lstm_75/while/NoOp*
T0*
_output_shapes
: 
lstm_75/while/Identity_1Identity.lstm_75_while_lstm_75_while_maximum_iterations^lstm_75/while/NoOp*
T0*
_output_shapes
: q
lstm_75/while/Identity_2Identitylstm_75/while/add:z:0^lstm_75/while/NoOp*
T0*
_output_shapes
: 
lstm_75/while/Identity_3IdentityBlstm_75/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_75/while/NoOp*
T0*
_output_shapes
: 
lstm_75/while/Identity_4Identity$lstm_75/while/lstm_cell_75/mul_2:z:0^lstm_75/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_75/while/Identity_5Identity$lstm_75/while/lstm_cell_75/add_1:z:0^lstm_75/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
lstm_75/while/NoOpNoOp2^lstm_75/while/lstm_cell_75/BiasAdd/ReadVariableOp1^lstm_75/while/lstm_cell_75/MatMul/ReadVariableOp3^lstm_75/while/lstm_cell_75/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_75_while_identitylstm_75/while/Identity:output:0"=
lstm_75_while_identity_1!lstm_75/while/Identity_1:output:0"=
lstm_75_while_identity_2!lstm_75/while/Identity_2:output:0"=
lstm_75_while_identity_3!lstm_75/while/Identity_3:output:0"=
lstm_75_while_identity_4!lstm_75/while/Identity_4:output:0"=
lstm_75_while_identity_5!lstm_75/while/Identity_5:output:0"P
%lstm_75_while_lstm_75_strided_slice_1'lstm_75_while_lstm_75_strided_slice_1_0"z
:lstm_75_while_lstm_cell_75_biasadd_readvariableop_resource<lstm_75_while_lstm_cell_75_biasadd_readvariableop_resource_0"|
;lstm_75_while_lstm_cell_75_matmul_1_readvariableop_resource=lstm_75_while_lstm_cell_75_matmul_1_readvariableop_resource_0"x
9lstm_75_while_lstm_cell_75_matmul_readvariableop_resource;lstm_75_while_lstm_cell_75_matmul_readvariableop_resource_0"È
alstm_75_while_tensorarrayv2read_tensorlistgetitem_lstm_75_tensorarrayunstack_tensorlistfromtensorclstm_75_while_tensorarrayv2read_tensorlistgetitem_lstm_75_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_75/while/lstm_cell_75/BiasAdd/ReadVariableOp1lstm_75/while/lstm_cell_75/BiasAdd/ReadVariableOp2d
0lstm_75/while/lstm_cell_75/MatMul/ReadVariableOp0lstm_75/while/lstm_cell_75/MatMul/ReadVariableOp2h
2lstm_75/while/lstm_cell_75/MatMul_1/ReadVariableOp2lstm_75/while/lstm_cell_75/MatMul_1/ReadVariableOp: 
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
Ý
d
F__inference_dropout_8_layer_call_and_return_conditional_losses_3247755

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
I__inference_lstm_cell_75_layer_call_and_return_conditional_losses_3247866

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


e
F__inference_dropout_8_layer_call_and_return_conditional_losses_3247767

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

¹
)__inference_lstm_75_layer_call_fn_3247138
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
D__inference_lstm_75_layer_call_and_return_conditional_losses_3246225p
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
Æ
_
3__inference_weighted_layer_33_layer_call_fn_3247792
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
N__inference_weighted_layer_33_layer_call_and_return_conditional_losses_3246424d
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
¨K
³
!ma_mod_lstm_75_while_body_3245785:
6ma_mod_lstm_75_while_ma_mod_lstm_75_while_loop_counter@
<ma_mod_lstm_75_while_ma_mod_lstm_75_while_maximum_iterations$
 ma_mod_lstm_75_while_placeholder&
"ma_mod_lstm_75_while_placeholder_1&
"ma_mod_lstm_75_while_placeholder_2&
"ma_mod_lstm_75_while_placeholder_39
5ma_mod_lstm_75_while_ma_mod_lstm_75_strided_slice_1_0u
qma_mod_lstm_75_while_tensorarrayv2read_tensorlistgetitem_ma_mod_lstm_75_tensorarrayunstack_tensorlistfromtensor_0U
Bma_mod_lstm_75_while_lstm_cell_75_matmul_readvariableop_resource_0:	X
Dma_mod_lstm_75_while_lstm_cell_75_matmul_1_readvariableop_resource_0:
R
Cma_mod_lstm_75_while_lstm_cell_75_biasadd_readvariableop_resource_0:	!
ma_mod_lstm_75_while_identity#
ma_mod_lstm_75_while_identity_1#
ma_mod_lstm_75_while_identity_2#
ma_mod_lstm_75_while_identity_3#
ma_mod_lstm_75_while_identity_4#
ma_mod_lstm_75_while_identity_57
3ma_mod_lstm_75_while_ma_mod_lstm_75_strided_slice_1s
oma_mod_lstm_75_while_tensorarrayv2read_tensorlistgetitem_ma_mod_lstm_75_tensorarrayunstack_tensorlistfromtensorS
@ma_mod_lstm_75_while_lstm_cell_75_matmul_readvariableop_resource:	V
Bma_mod_lstm_75_while_lstm_cell_75_matmul_1_readvariableop_resource:
P
Ama_mod_lstm_75_while_lstm_cell_75_biasadd_readvariableop_resource:	¢8ma_mod/lstm_75/while/lstm_cell_75/BiasAdd/ReadVariableOp¢7ma_mod/lstm_75/while/lstm_cell_75/MatMul/ReadVariableOp¢9ma_mod/lstm_75/while/lstm_cell_75/MatMul_1/ReadVariableOp
Fma_mod/lstm_75/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ñ
8ma_mod/lstm_75/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqma_mod_lstm_75_while_tensorarrayv2read_tensorlistgetitem_ma_mod_lstm_75_tensorarrayunstack_tensorlistfromtensor_0 ma_mod_lstm_75_while_placeholderOma_mod/lstm_75/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0»
7ma_mod/lstm_75/while/lstm_cell_75/MatMul/ReadVariableOpReadVariableOpBma_mod_lstm_75_while_lstm_cell_75_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0ç
(ma_mod/lstm_75/while/lstm_cell_75/MatMulMatMul?ma_mod/lstm_75/while/TensorArrayV2Read/TensorListGetItem:item:0?ma_mod/lstm_75/while/lstm_cell_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
9ma_mod/lstm_75/while/lstm_cell_75/MatMul_1/ReadVariableOpReadVariableOpDma_mod_lstm_75_while_lstm_cell_75_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Î
*ma_mod/lstm_75/while/lstm_cell_75/MatMul_1MatMul"ma_mod_lstm_75_while_placeholder_2Ama_mod/lstm_75/while/lstm_cell_75/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
%ma_mod/lstm_75/while/lstm_cell_75/addAddV22ma_mod/lstm_75/while/lstm_cell_75/MatMul:product:04ma_mod/lstm_75/while/lstm_cell_75/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
8ma_mod/lstm_75/while/lstm_cell_75/BiasAdd/ReadVariableOpReadVariableOpCma_mod_lstm_75_while_lstm_cell_75_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ô
)ma_mod/lstm_75/while/lstm_cell_75/BiasAddBiasAdd)ma_mod/lstm_75/while/lstm_cell_75/add:z:0@ma_mod/lstm_75/while/lstm_cell_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
1ma_mod/lstm_75/while/lstm_cell_75/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
'ma_mod/lstm_75/while/lstm_cell_75/splitSplit:ma_mod/lstm_75/while/lstm_cell_75/split/split_dim:output:02ma_mod/lstm_75/while/lstm_cell_75/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
)ma_mod/lstm_75/while/lstm_cell_75/SigmoidSigmoid0ma_mod/lstm_75/while/lstm_cell_75/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+ma_mod/lstm_75/while/lstm_cell_75/Sigmoid_1Sigmoid0ma_mod/lstm_75/while/lstm_cell_75/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
%ma_mod/lstm_75/while/lstm_cell_75/mulMul/ma_mod/lstm_75/while/lstm_cell_75/Sigmoid_1:y:0"ma_mod_lstm_75_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&ma_mod/lstm_75/while/lstm_cell_75/ReluRelu0ma_mod/lstm_75/while/lstm_cell_75/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
'ma_mod/lstm_75/while/lstm_cell_75/mul_1Mul-ma_mod/lstm_75/while/lstm_cell_75/Sigmoid:y:04ma_mod/lstm_75/while/lstm_cell_75/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
'ma_mod/lstm_75/while/lstm_cell_75/add_1AddV2)ma_mod/lstm_75/while/lstm_cell_75/mul:z:0+ma_mod/lstm_75/while/lstm_cell_75/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+ma_mod/lstm_75/while/lstm_cell_75/Sigmoid_2Sigmoid0ma_mod/lstm_75/while/lstm_cell_75/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(ma_mod/lstm_75/while/lstm_cell_75/Relu_1Relu+ma_mod/lstm_75/while/lstm_cell_75/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
'ma_mod/lstm_75/while/lstm_cell_75/mul_2Mul/ma_mod/lstm_75/while/lstm_cell_75/Sigmoid_2:y:06ma_mod/lstm_75/while/lstm_cell_75/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
?ma_mod/lstm_75/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ©
9ma_mod/lstm_75/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"ma_mod_lstm_75_while_placeholder_1Hma_mod/lstm_75/while/TensorArrayV2Write/TensorListSetItem/index:output:0+ma_mod/lstm_75/while/lstm_cell_75/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒ\
ma_mod/lstm_75/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
ma_mod/lstm_75/while/addAddV2 ma_mod_lstm_75_while_placeholder#ma_mod/lstm_75/while/add/y:output:0*
T0*
_output_shapes
: ^
ma_mod/lstm_75/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :£
ma_mod/lstm_75/while/add_1AddV26ma_mod_lstm_75_while_ma_mod_lstm_75_while_loop_counter%ma_mod/lstm_75/while/add_1/y:output:0*
T0*
_output_shapes
: 
ma_mod/lstm_75/while/IdentityIdentityma_mod/lstm_75/while/add_1:z:0^ma_mod/lstm_75/while/NoOp*
T0*
_output_shapes
: ¦
ma_mod/lstm_75/while/Identity_1Identity<ma_mod_lstm_75_while_ma_mod_lstm_75_while_maximum_iterations^ma_mod/lstm_75/while/NoOp*
T0*
_output_shapes
: 
ma_mod/lstm_75/while/Identity_2Identityma_mod/lstm_75/while/add:z:0^ma_mod/lstm_75/while/NoOp*
T0*
_output_shapes
: ³
ma_mod/lstm_75/while/Identity_3IdentityIma_mod/lstm_75/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^ma_mod/lstm_75/while/NoOp*
T0*
_output_shapes
: §
ma_mod/lstm_75/while/Identity_4Identity+ma_mod/lstm_75/while/lstm_cell_75/mul_2:z:0^ma_mod/lstm_75/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
ma_mod/lstm_75/while/Identity_5Identity+ma_mod/lstm_75/while/lstm_cell_75/add_1:z:0^ma_mod/lstm_75/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ma_mod/lstm_75/while/NoOpNoOp9^ma_mod/lstm_75/while/lstm_cell_75/BiasAdd/ReadVariableOp8^ma_mod/lstm_75/while/lstm_cell_75/MatMul/ReadVariableOp:^ma_mod/lstm_75/while/lstm_cell_75/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "G
ma_mod_lstm_75_while_identity&ma_mod/lstm_75/while/Identity:output:0"K
ma_mod_lstm_75_while_identity_1(ma_mod/lstm_75/while/Identity_1:output:0"K
ma_mod_lstm_75_while_identity_2(ma_mod/lstm_75/while/Identity_2:output:0"K
ma_mod_lstm_75_while_identity_3(ma_mod/lstm_75/while/Identity_3:output:0"K
ma_mod_lstm_75_while_identity_4(ma_mod/lstm_75/while/Identity_4:output:0"K
ma_mod_lstm_75_while_identity_5(ma_mod/lstm_75/while/Identity_5:output:0"
Ama_mod_lstm_75_while_lstm_cell_75_biasadd_readvariableop_resourceCma_mod_lstm_75_while_lstm_cell_75_biasadd_readvariableop_resource_0"
Bma_mod_lstm_75_while_lstm_cell_75_matmul_1_readvariableop_resourceDma_mod_lstm_75_while_lstm_cell_75_matmul_1_readvariableop_resource_0"
@ma_mod_lstm_75_while_lstm_cell_75_matmul_readvariableop_resourceBma_mod_lstm_75_while_lstm_cell_75_matmul_readvariableop_resource_0"l
3ma_mod_lstm_75_while_ma_mod_lstm_75_strided_slice_15ma_mod_lstm_75_while_ma_mod_lstm_75_strided_slice_1_0"ä
oma_mod_lstm_75_while_tensorarrayv2read_tensorlistgetitem_ma_mod_lstm_75_tensorarrayunstack_tensorlistfromtensorqma_mod_lstm_75_while_tensorarrayv2read_tensorlistgetitem_ma_mod_lstm_75_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2t
8ma_mod/lstm_75/while/lstm_cell_75/BiasAdd/ReadVariableOp8ma_mod/lstm_75/while/lstm_cell_75/BiasAdd/ReadVariableOp2r
7ma_mod/lstm_75/while/lstm_cell_75/MatMul/ReadVariableOp7ma_mod/lstm_75/while/lstm_cell_75/MatMul/ReadVariableOp2v
9ma_mod/lstm_75/while/lstm_cell_75/MatMul_1/ReadVariableOp9ma_mod/lstm_75/while/lstm_cell_75/MatMul_1/ReadVariableOp: 
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
"__inference__wrapped_model_3245880
input_48M
:ma_mod_lstm_75_lstm_cell_75_matmul_readvariableop_resource:	P
<ma_mod_lstm_75_lstm_cell_75_matmul_1_readvariableop_resource:
J
;ma_mod_lstm_75_lstm_cell_75_biasadd_readvariableop_resource:	A
.ma_mod_dense_72_matmul_readvariableop_resource:	
=
/ma_mod_dense_72_biasadd_readvariableop_resource:

identity¢&ma_mod/dense_72/BiasAdd/ReadVariableOp¢%ma_mod/dense_72/MatMul/ReadVariableOp¢2ma_mod/lstm_75/lstm_cell_75/BiasAdd/ReadVariableOp¢1ma_mod/lstm_75/lstm_cell_75/MatMul/ReadVariableOp¢3ma_mod/lstm_75/lstm_cell_75/MatMul_1/ReadVariableOp¢ma_mod/lstm_75/whileL
ma_mod/lstm_75/ShapeShapeinput_48*
T0*
_output_shapes
:l
"ma_mod/lstm_75/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$ma_mod/lstm_75/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$ma_mod/lstm_75/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ma_mod/lstm_75/strided_sliceStridedSlicema_mod/lstm_75/Shape:output:0+ma_mod/lstm_75/strided_slice/stack:output:0-ma_mod/lstm_75/strided_slice/stack_1:output:0-ma_mod/lstm_75/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
ma_mod/lstm_75/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B : 
ma_mod/lstm_75/zeros/packedPack%ma_mod/lstm_75/strided_slice:output:0&ma_mod/lstm_75/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:_
ma_mod/lstm_75/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
ma_mod/lstm_75/zerosFill$ma_mod/lstm_75/zeros/packed:output:0#ma_mod/lstm_75/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
ma_mod/lstm_75/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¤
ma_mod/lstm_75/zeros_1/packedPack%ma_mod/lstm_75/strided_slice:output:0(ma_mod/lstm_75/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:a
ma_mod/lstm_75/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *     
ma_mod/lstm_75/zeros_1Fill&ma_mod/lstm_75/zeros_1/packed:output:0%ma_mod/lstm_75/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
ma_mod/lstm_75/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
ma_mod/lstm_75/transpose	Transposeinput_48&ma_mod/lstm_75/transpose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿb
ma_mod/lstm_75/Shape_1Shapema_mod/lstm_75/transpose:y:0*
T0*
_output_shapes
:n
$ma_mod/lstm_75/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&ma_mod/lstm_75/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&ma_mod/lstm_75/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
ma_mod/lstm_75/strided_slice_1StridedSlicema_mod/lstm_75/Shape_1:output:0-ma_mod/lstm_75/strided_slice_1/stack:output:0/ma_mod/lstm_75/strided_slice_1/stack_1:output:0/ma_mod/lstm_75/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
*ma_mod/lstm_75/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿá
ma_mod/lstm_75/TensorArrayV2TensorListReserve3ma_mod/lstm_75/TensorArrayV2/element_shape:output:0'ma_mod/lstm_75/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Dma_mod/lstm_75/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
6ma_mod/lstm_75/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorma_mod/lstm_75/transpose:y:0Mma_mod/lstm_75/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$ma_mod/lstm_75/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&ma_mod/lstm_75/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&ma_mod/lstm_75/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:´
ma_mod/lstm_75/strided_slice_2StridedSlicema_mod/lstm_75/transpose:y:0-ma_mod/lstm_75/strided_slice_2/stack:output:0/ma_mod/lstm_75/strided_slice_2/stack_1:output:0/ma_mod/lstm_75/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask­
1ma_mod/lstm_75/lstm_cell_75/MatMul/ReadVariableOpReadVariableOp:ma_mod_lstm_75_lstm_cell_75_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Ã
"ma_mod/lstm_75/lstm_cell_75/MatMulMatMul'ma_mod/lstm_75/strided_slice_2:output:09ma_mod/lstm_75/lstm_cell_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
3ma_mod/lstm_75/lstm_cell_75/MatMul_1/ReadVariableOpReadVariableOp<ma_mod_lstm_75_lstm_cell_75_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0½
$ma_mod/lstm_75/lstm_cell_75/MatMul_1MatMulma_mod/lstm_75/zeros:output:0;ma_mod/lstm_75/lstm_cell_75/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
ma_mod/lstm_75/lstm_cell_75/addAddV2,ma_mod/lstm_75/lstm_cell_75/MatMul:product:0.ma_mod/lstm_75/lstm_cell_75/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
2ma_mod/lstm_75/lstm_cell_75/BiasAdd/ReadVariableOpReadVariableOp;ma_mod_lstm_75_lstm_cell_75_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Â
#ma_mod/lstm_75/lstm_cell_75/BiasAddBiasAdd#ma_mod/lstm_75/lstm_cell_75/add:z:0:ma_mod/lstm_75/lstm_cell_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
+ma_mod/lstm_75/lstm_cell_75/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!ma_mod/lstm_75/lstm_cell_75/splitSplit4ma_mod/lstm_75/lstm_cell_75/split/split_dim:output:0,ma_mod/lstm_75/lstm_cell_75/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
#ma_mod/lstm_75/lstm_cell_75/SigmoidSigmoid*ma_mod/lstm_75/lstm_cell_75/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%ma_mod/lstm_75/lstm_cell_75/Sigmoid_1Sigmoid*ma_mod/lstm_75/lstm_cell_75/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
ma_mod/lstm_75/lstm_cell_75/mulMul)ma_mod/lstm_75/lstm_cell_75/Sigmoid_1:y:0ma_mod/lstm_75/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 ma_mod/lstm_75/lstm_cell_75/ReluRelu*ma_mod/lstm_75/lstm_cell_75/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
!ma_mod/lstm_75/lstm_cell_75/mul_1Mul'ma_mod/lstm_75/lstm_cell_75/Sigmoid:y:0.ma_mod/lstm_75/lstm_cell_75/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
!ma_mod/lstm_75/lstm_cell_75/add_1AddV2#ma_mod/lstm_75/lstm_cell_75/mul:z:0%ma_mod/lstm_75/lstm_cell_75/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%ma_mod/lstm_75/lstm_cell_75/Sigmoid_2Sigmoid*ma_mod/lstm_75/lstm_cell_75/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"ma_mod/lstm_75/lstm_cell_75/Relu_1Relu%ma_mod/lstm_75/lstm_cell_75/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
!ma_mod/lstm_75/lstm_cell_75/mul_2Mul)ma_mod/lstm_75/lstm_cell_75/Sigmoid_2:y:00ma_mod/lstm_75/lstm_cell_75/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
,ma_mod/lstm_75/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   m
+ma_mod/lstm_75/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ò
ma_mod/lstm_75/TensorArrayV2_1TensorListReserve5ma_mod/lstm_75/TensorArrayV2_1/element_shape:output:04ma_mod/lstm_75/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒU
ma_mod/lstm_75/timeConst*
_output_shapes
: *
dtype0*
value	B : r
'ma_mod/lstm_75/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿc
!ma_mod/lstm_75/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ú
ma_mod/lstm_75/whileWhile*ma_mod/lstm_75/while/loop_counter:output:00ma_mod/lstm_75/while/maximum_iterations:output:0ma_mod/lstm_75/time:output:0'ma_mod/lstm_75/TensorArrayV2_1:handle:0ma_mod/lstm_75/zeros:output:0ma_mod/lstm_75/zeros_1:output:0'ma_mod/lstm_75/strided_slice_1:output:0Fma_mod/lstm_75/TensorArrayUnstack/TensorListFromTensor:output_handle:0:ma_mod_lstm_75_lstm_cell_75_matmul_readvariableop_resource<ma_mod_lstm_75_lstm_cell_75_matmul_1_readvariableop_resource;ma_mod_lstm_75_lstm_cell_75_biasadd_readvariableop_resource*
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
!ma_mod_lstm_75_while_body_3245785*-
cond%R#
!ma_mod_lstm_75_while_cond_3245784*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
?ma_mod/lstm_75/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
1ma_mod/lstm_75/TensorArrayV2Stack/TensorListStackTensorListStackma_mod/lstm_75/while:output:3Hma_mod/lstm_75/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsw
$ma_mod/lstm_75/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿp
&ma_mod/lstm_75/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&ma_mod/lstm_75/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
ma_mod/lstm_75/strided_slice_3StridedSlice:ma_mod/lstm_75/TensorArrayV2Stack/TensorListStack:tensor:0-ma_mod/lstm_75/strided_slice_3/stack:output:0/ma_mod/lstm_75/strided_slice_3/stack_1:output:0/ma_mod/lstm_75/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskt
ma_mod/lstm_75/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ä
ma_mod/lstm_75/transpose_1	Transpose:ma_mod/lstm_75/TensorArrayV2Stack/TensorListStack:tensor:0(ma_mod/lstm_75/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
ma_mod/lstm_75/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
ma_mod/dropout_8/IdentityIdentity'ma_mod/lstm_75/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%ma_mod/dense_72/MatMul/ReadVariableOpReadVariableOp.ma_mod_dense_72_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0¥
ma_mod/dense_72/MatMulMatMul"ma_mod/dropout_8/Identity:output:0-ma_mod/dense_72/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&ma_mod/dense_72/BiasAdd/ReadVariableOpReadVariableOp/ma_mod_dense_72_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0¦
ma_mod/dense_72/BiasAddBiasAdd ma_mod/dense_72/MatMul:product:0.ma_mod/dense_72/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
&ma_mod/weighted_layer_33/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   °
 ma_mod/weighted_layer_33/ReshapeReshape ma_mod/dense_72/BiasAdd:output:0/ma_mod/weighted_layer_33/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ma_mod/weighted_layer_33/MulMulinput_48)ma_mod/weighted_layer_33/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
IdentityIdentity ma_mod/weighted_layer_33/Mul:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Í
NoOpNoOp'^ma_mod/dense_72/BiasAdd/ReadVariableOp&^ma_mod/dense_72/MatMul/ReadVariableOp3^ma_mod/lstm_75/lstm_cell_75/BiasAdd/ReadVariableOp2^ma_mod/lstm_75/lstm_cell_75/MatMul/ReadVariableOp4^ma_mod/lstm_75/lstm_cell_75/MatMul_1/ReadVariableOp^ma_mod/lstm_75/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ
: : : : : 2P
&ma_mod/dense_72/BiasAdd/ReadVariableOp&ma_mod/dense_72/BiasAdd/ReadVariableOp2N
%ma_mod/dense_72/MatMul/ReadVariableOp%ma_mod/dense_72/MatMul/ReadVariableOp2h
2ma_mod/lstm_75/lstm_cell_75/BiasAdd/ReadVariableOp2ma_mod/lstm_75/lstm_cell_75/BiasAdd/ReadVariableOp2f
1ma_mod/lstm_75/lstm_cell_75/MatMul/ReadVariableOp1ma_mod/lstm_75/lstm_cell_75/MatMul/ReadVariableOp2j
3ma_mod/lstm_75/lstm_cell_75/MatMul_1/ReadVariableOp3ma_mod/lstm_75/lstm_cell_75/MatMul_1/ReadVariableOp2,
ma_mod/lstm_75/whilema_mod/lstm_75/while:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
input_48
Ý
d
F__inference_dropout_8_layer_call_and_return_conditional_losses_3246398

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
¿9
Ó
while_body_3246553
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_75_matmul_readvariableop_resource_0:	I
5while_lstm_cell_75_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_75_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_75_matmul_readvariableop_resource:	G
3while_lstm_cell_75_matmul_1_readvariableop_resource:
A
2while_lstm_cell_75_biasadd_readvariableop_resource:	¢)while/lstm_cell_75/BiasAdd/ReadVariableOp¢(while/lstm_cell_75/MatMul/ReadVariableOp¢*while/lstm_cell_75/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_75/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_75_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_75/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_75/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_75_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_75/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_75/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/addAddV2#while/lstm_cell_75/MatMul:product:0%while/lstm_cell_75/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_75/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_75_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_75/BiasAddBiasAddwhile/lstm_cell_75/add:z:01while/lstm_cell_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_75/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_75/splitSplit+while/lstm_cell_75/split/split_dim:output:0#while/lstm_cell_75/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_75/SigmoidSigmoid!while/lstm_cell_75/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_75/Sigmoid_1Sigmoid!while/lstm_cell_75/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/mulMul while/lstm_cell_75/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_75/ReluRelu!while/lstm_cell_75/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/mul_1Mulwhile/lstm_cell_75/Sigmoid:y:0%while/lstm_cell_75/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/add_1AddV2while/lstm_cell_75/mul:z:0while/lstm_cell_75/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_75/Sigmoid_2Sigmoid!while/lstm_cell_75/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_75/Relu_1Reluwhile/lstm_cell_75/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/mul_2Mul while/lstm_cell_75/Sigmoid_2:y:0'while/lstm_cell_75/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_75/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_75/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_75/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_75/BiasAdd/ReadVariableOp)^while/lstm_cell_75/MatMul/ReadVariableOp+^while/lstm_cell_75/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_75_biasadd_readvariableop_resource4while_lstm_cell_75_biasadd_readvariableop_resource_0"l
3while_lstm_cell_75_matmul_1_readvariableop_resource5while_lstm_cell_75_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_75_matmul_readvariableop_resource3while_lstm_cell_75_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_75/BiasAdd/ReadVariableOp)while/lstm_cell_75/BiasAdd/ReadVariableOp2T
(while/lstm_cell_75/MatMul/ReadVariableOp(while/lstm_cell_75/MatMul/ReadVariableOp2X
*while/lstm_cell_75/MatMul_1/ReadVariableOp*while/lstm_cell_75/MatMul_1/ReadVariableOp: 
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
.__inference_lstm_cell_75_layer_call_fn_3247834

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
I__inference_lstm_cell_75_layer_call_and_return_conditional_losses_3246095p
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
ü
·
)__inference_lstm_75_layer_call_fn_3247160

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
D__inference_lstm_75_layer_call_and_return_conditional_losses_3246638p
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
ÇK

D__inference_lstm_75_layer_call_and_return_conditional_losses_3247740

inputs>
+lstm_cell_75_matmul_readvariableop_resource:	A
-lstm_cell_75_matmul_1_readvariableop_resource:
;
,lstm_cell_75_biasadd_readvariableop_resource:	
identity¢#lstm_cell_75/BiasAdd/ReadVariableOp¢"lstm_cell_75/MatMul/ReadVariableOp¢$lstm_cell_75/MatMul_1/ReadVariableOp¢while;
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
"lstm_cell_75/MatMul/ReadVariableOpReadVariableOp+lstm_cell_75_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_75/MatMulMatMulstrided_slice_2:output:0*lstm_cell_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_75/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_75_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_75/MatMul_1MatMulzeros:output:0,lstm_cell_75/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_75/addAddV2lstm_cell_75/MatMul:product:0lstm_cell_75/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_75/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_75_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_75/BiasAddBiasAddlstm_cell_75/add:z:0+lstm_cell_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_75/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_75/splitSplit%lstm_cell_75/split/split_dim:output:0lstm_cell_75/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_75/SigmoidSigmoidlstm_cell_75/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_75/Sigmoid_1Sigmoidlstm_cell_75/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_75/mulMullstm_cell_75/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_75/ReluRelulstm_cell_75/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_75/mul_1Mullstm_cell_75/Sigmoid:y:0lstm_cell_75/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_75/add_1AddV2lstm_cell_75/mul:z:0lstm_cell_75/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_75/Sigmoid_2Sigmoidlstm_cell_75/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_75/Relu_1Relulstm_cell_75/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_75/mul_2Mullstm_cell_75/Sigmoid_2:y:0!lstm_cell_75/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_75_matmul_readvariableop_resource-lstm_cell_75_matmul_1_readvariableop_resource,lstm_cell_75_biasadd_readvariableop_resource*
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
while_body_3247655*
condR
while_cond_3247654*M
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
NoOpNoOp$^lstm_cell_75/BiasAdd/ReadVariableOp#^lstm_cell_75/MatMul/ReadVariableOp%^lstm_cell_75/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 2J
#lstm_cell_75/BiasAdd/ReadVariableOp#lstm_cell_75/BiasAdd/ReadVariableOp2H
"lstm_cell_75/MatMul/ReadVariableOp"lstm_cell_75/MatMul/ReadVariableOp2L
$lstm_cell_75/MatMul_1/ReadVariableOp$lstm_cell_75/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs


e
F__inference_dropout_8_layer_call_and_return_conditional_losses_3246477

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
I__inference_lstm_cell_75_layer_call_and_return_conditional_losses_3246095

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
while_cond_3246299
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3246299___redundant_placeholder05
1while_while_cond_3246299___redundant_placeholder15
1while_while_cond_3246299___redundant_placeholder25
1while_while_cond_3246299___redundant_placeholder3
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
ÇK

D__inference_lstm_75_layer_call_and_return_conditional_losses_3246385

inputs>
+lstm_cell_75_matmul_readvariableop_resource:	A
-lstm_cell_75_matmul_1_readvariableop_resource:
;
,lstm_cell_75_biasadd_readvariableop_resource:	
identity¢#lstm_cell_75/BiasAdd/ReadVariableOp¢"lstm_cell_75/MatMul/ReadVariableOp¢$lstm_cell_75/MatMul_1/ReadVariableOp¢while;
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
"lstm_cell_75/MatMul/ReadVariableOpReadVariableOp+lstm_cell_75_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_75/MatMulMatMulstrided_slice_2:output:0*lstm_cell_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_75/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_75_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_75/MatMul_1MatMulzeros:output:0,lstm_cell_75/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_75/addAddV2lstm_cell_75/MatMul:product:0lstm_cell_75/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_75/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_75_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_75/BiasAddBiasAddlstm_cell_75/add:z:0+lstm_cell_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_75/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_75/splitSplit%lstm_cell_75/split/split_dim:output:0lstm_cell_75/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_75/SigmoidSigmoidlstm_cell_75/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_75/Sigmoid_1Sigmoidlstm_cell_75/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_75/mulMullstm_cell_75/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_75/ReluRelulstm_cell_75/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_75/mul_1Mullstm_cell_75/Sigmoid:y:0lstm_cell_75/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_75/add_1AddV2lstm_cell_75/mul:z:0lstm_cell_75/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_75/Sigmoid_2Sigmoidlstm_cell_75/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_75/Relu_1Relulstm_cell_75/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_75/mul_2Mullstm_cell_75/Sigmoid_2:y:0!lstm_cell_75/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_75_matmul_readvariableop_resource-lstm_cell_75_matmul_1_readvariableop_resource,lstm_cell_75_biasadd_readvariableop_resource*
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
while_body_3246300*
condR
while_cond_3246299*M
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
NoOpNoOp$^lstm_cell_75/BiasAdd/ReadVariableOp#^lstm_cell_75/MatMul/ReadVariableOp%^lstm_cell_75/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 2J
#lstm_cell_75/BiasAdd/ReadVariableOp#lstm_cell_75/BiasAdd/ReadVariableOp2H
"lstm_cell_75/MatMul/ReadVariableOp"lstm_cell_75/MatMul/ReadVariableOp2L
$lstm_cell_75/MatMul_1/ReadVariableOp$lstm_cell_75/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
C
Ó

lstm_75_while_body_3246859,
(lstm_75_while_lstm_75_while_loop_counter2
.lstm_75_while_lstm_75_while_maximum_iterations
lstm_75_while_placeholder
lstm_75_while_placeholder_1
lstm_75_while_placeholder_2
lstm_75_while_placeholder_3+
'lstm_75_while_lstm_75_strided_slice_1_0g
clstm_75_while_tensorarrayv2read_tensorlistgetitem_lstm_75_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_75_while_lstm_cell_75_matmul_readvariableop_resource_0:	Q
=lstm_75_while_lstm_cell_75_matmul_1_readvariableop_resource_0:
K
<lstm_75_while_lstm_cell_75_biasadd_readvariableop_resource_0:	
lstm_75_while_identity
lstm_75_while_identity_1
lstm_75_while_identity_2
lstm_75_while_identity_3
lstm_75_while_identity_4
lstm_75_while_identity_5)
%lstm_75_while_lstm_75_strided_slice_1e
alstm_75_while_tensorarrayv2read_tensorlistgetitem_lstm_75_tensorarrayunstack_tensorlistfromtensorL
9lstm_75_while_lstm_cell_75_matmul_readvariableop_resource:	O
;lstm_75_while_lstm_cell_75_matmul_1_readvariableop_resource:
I
:lstm_75_while_lstm_cell_75_biasadd_readvariableop_resource:	¢1lstm_75/while/lstm_cell_75/BiasAdd/ReadVariableOp¢0lstm_75/while/lstm_cell_75/MatMul/ReadVariableOp¢2lstm_75/while/lstm_cell_75/MatMul_1/ReadVariableOp
?lstm_75/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Î
1lstm_75/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_75_while_tensorarrayv2read_tensorlistgetitem_lstm_75_tensorarrayunstack_tensorlistfromtensor_0lstm_75_while_placeholderHlstm_75/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0­
0lstm_75/while/lstm_cell_75/MatMul/ReadVariableOpReadVariableOp;lstm_75_while_lstm_cell_75_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ò
!lstm_75/while/lstm_cell_75/MatMulMatMul8lstm_75/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_75/while/lstm_cell_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
2lstm_75/while/lstm_cell_75/MatMul_1/ReadVariableOpReadVariableOp=lstm_75_while_lstm_cell_75_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¹
#lstm_75/while/lstm_cell_75/MatMul_1MatMullstm_75_while_placeholder_2:lstm_75/while/lstm_cell_75/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm_75/while/lstm_cell_75/addAddV2+lstm_75/while/lstm_cell_75/MatMul:product:0-lstm_75/while/lstm_cell_75/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
1lstm_75/while/lstm_cell_75/BiasAdd/ReadVariableOpReadVariableOp<lstm_75_while_lstm_cell_75_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¿
"lstm_75/while/lstm_cell_75/BiasAddBiasAdd"lstm_75/while/lstm_cell_75/add:z:09lstm_75/while/lstm_cell_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*lstm_75/while/lstm_cell_75/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_75/while/lstm_cell_75/splitSplit3lstm_75/while/lstm_cell_75/split/split_dim:output:0+lstm_75/while/lstm_cell_75/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"lstm_75/while/lstm_cell_75/SigmoidSigmoid)lstm_75/while/lstm_cell_75/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_75/while/lstm_cell_75/Sigmoid_1Sigmoid)lstm_75/while/lstm_cell_75/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_75/while/lstm_cell_75/mulMul(lstm_75/while/lstm_cell_75/Sigmoid_1:y:0lstm_75_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_75/while/lstm_cell_75/ReluRelu)lstm_75/while/lstm_cell_75/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
 lstm_75/while/lstm_cell_75/mul_1Mul&lstm_75/while/lstm_cell_75/Sigmoid:y:0-lstm_75/while/lstm_cell_75/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
 lstm_75/while/lstm_cell_75/add_1AddV2"lstm_75/while/lstm_cell_75/mul:z:0$lstm_75/while/lstm_cell_75/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_75/while/lstm_cell_75/Sigmoid_2Sigmoid)lstm_75/while/lstm_cell_75/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_75/while/lstm_cell_75/Relu_1Relu$lstm_75/while/lstm_cell_75/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
 lstm_75/while/lstm_cell_75/mul_2Mul(lstm_75/while/lstm_cell_75/Sigmoid_2:y:0/lstm_75/while/lstm_cell_75/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8lstm_75/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
2lstm_75/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_75_while_placeholder_1Alstm_75/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_75/while/lstm_cell_75/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_75/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_75/while/addAddV2lstm_75_while_placeholderlstm_75/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_75/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_75/while/add_1AddV2(lstm_75_while_lstm_75_while_loop_counterlstm_75/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_75/while/IdentityIdentitylstm_75/while/add_1:z:0^lstm_75/while/NoOp*
T0*
_output_shapes
: 
lstm_75/while/Identity_1Identity.lstm_75_while_lstm_75_while_maximum_iterations^lstm_75/while/NoOp*
T0*
_output_shapes
: q
lstm_75/while/Identity_2Identitylstm_75/while/add:z:0^lstm_75/while/NoOp*
T0*
_output_shapes
: 
lstm_75/while/Identity_3IdentityBlstm_75/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_75/while/NoOp*
T0*
_output_shapes
: 
lstm_75/while/Identity_4Identity$lstm_75/while/lstm_cell_75/mul_2:z:0^lstm_75/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_75/while/Identity_5Identity$lstm_75/while/lstm_cell_75/add_1:z:0^lstm_75/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
lstm_75/while/NoOpNoOp2^lstm_75/while/lstm_cell_75/BiasAdd/ReadVariableOp1^lstm_75/while/lstm_cell_75/MatMul/ReadVariableOp3^lstm_75/while/lstm_cell_75/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_75_while_identitylstm_75/while/Identity:output:0"=
lstm_75_while_identity_1!lstm_75/while/Identity_1:output:0"=
lstm_75_while_identity_2!lstm_75/while/Identity_2:output:0"=
lstm_75_while_identity_3!lstm_75/while/Identity_3:output:0"=
lstm_75_while_identity_4!lstm_75/while/Identity_4:output:0"=
lstm_75_while_identity_5!lstm_75/while/Identity_5:output:0"P
%lstm_75_while_lstm_75_strided_slice_1'lstm_75_while_lstm_75_strided_slice_1_0"z
:lstm_75_while_lstm_cell_75_biasadd_readvariableop_resource<lstm_75_while_lstm_cell_75_biasadd_readvariableop_resource_0"|
;lstm_75_while_lstm_cell_75_matmul_1_readvariableop_resource=lstm_75_while_lstm_cell_75_matmul_1_readvariableop_resource_0"x
9lstm_75_while_lstm_cell_75_matmul_readvariableop_resource;lstm_75_while_lstm_cell_75_matmul_readvariableop_resource_0"È
alstm_75_while_tensorarrayv2read_tensorlistgetitem_lstm_75_tensorarrayunstack_tensorlistfromtensorclstm_75_while_tensorarrayv2read_tensorlistgetitem_lstm_75_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_75/while/lstm_cell_75/BiasAdd/ReadVariableOp1lstm_75/while/lstm_cell_75/BiasAdd/ReadVariableOp2d
0lstm_75/while/lstm_cell_75/MatMul/ReadVariableOp0lstm_75/while/lstm_cell_75/MatMul/ReadVariableOp2h
2lstm_75/while/lstm_cell_75/MatMul_1/ReadVariableOp2lstm_75/while/lstm_cell_75/MatMul_1/ReadVariableOp: 
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
Û
ï
(__inference_ma_mod_layer_call_fn_3246710
input_48
unknown:	
	unknown_0:

	unknown_1:	
	unknown_2:	

	unknown_3:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_48unknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
C__inference_ma_mod_layer_call_and_return_conditional_losses_3246682s
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
input_48


è
lstm_75_while_cond_3247013,
(lstm_75_while_lstm_75_while_loop_counter2
.lstm_75_while_lstm_75_while_maximum_iterations
lstm_75_while_placeholder
lstm_75_while_placeholder_1
lstm_75_while_placeholder_2
lstm_75_while_placeholder_3.
*lstm_75_while_less_lstm_75_strided_slice_1E
Alstm_75_while_lstm_75_while_cond_3247013___redundant_placeholder0E
Alstm_75_while_lstm_75_while_cond_3247013___redundant_placeholder1E
Alstm_75_while_lstm_75_while_cond_3247013___redundant_placeholder2E
Alstm_75_while_lstm_75_while_cond_3247013___redundant_placeholder3
lstm_75_while_identity

lstm_75/while/LessLesslstm_75_while_placeholder*lstm_75_while_less_lstm_75_strided_slice_1*
T0*
_output_shapes
: [
lstm_75/while/IdentityIdentitylstm_75/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_75_while_identitylstm_75/while/Identity:output:0*(
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
D__inference_lstm_75_layer_call_and_return_conditional_losses_3247450
inputs_0>
+lstm_cell_75_matmul_readvariableop_resource:	A
-lstm_cell_75_matmul_1_readvariableop_resource:
;
,lstm_cell_75_biasadd_readvariableop_resource:	
identity¢#lstm_cell_75/BiasAdd/ReadVariableOp¢"lstm_cell_75/MatMul/ReadVariableOp¢$lstm_cell_75/MatMul_1/ReadVariableOp¢while=
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
"lstm_cell_75/MatMul/ReadVariableOpReadVariableOp+lstm_cell_75_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_75/MatMulMatMulstrided_slice_2:output:0*lstm_cell_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_75/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_75_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_75/MatMul_1MatMulzeros:output:0,lstm_cell_75/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_75/addAddV2lstm_cell_75/MatMul:product:0lstm_cell_75/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_75/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_75_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_75/BiasAddBiasAddlstm_cell_75/add:z:0+lstm_cell_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_75/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_75/splitSplit%lstm_cell_75/split/split_dim:output:0lstm_cell_75/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_75/SigmoidSigmoidlstm_cell_75/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_75/Sigmoid_1Sigmoidlstm_cell_75/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_75/mulMullstm_cell_75/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_75/ReluRelulstm_cell_75/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_75/mul_1Mullstm_cell_75/Sigmoid:y:0lstm_cell_75/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_75/add_1AddV2lstm_cell_75/mul:z:0lstm_cell_75/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_75/Sigmoid_2Sigmoidlstm_cell_75/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_75/Relu_1Relulstm_cell_75/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_75/mul_2Mullstm_cell_75/Sigmoid_2:y:0!lstm_cell_75/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_75_matmul_readvariableop_resource-lstm_cell_75_matmul_1_readvariableop_resource,lstm_cell_75_biasadd_readvariableop_resource*
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
while_body_3247365*
condR
while_cond_3247364*M
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
NoOpNoOp$^lstm_cell_75/BiasAdd/ReadVariableOp#^lstm_cell_75/MatMul/ReadVariableOp%^lstm_cell_75/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_75/BiasAdd/ReadVariableOp#lstm_cell_75/BiasAdd/ReadVariableOp2H
"lstm_cell_75/MatMul/ReadVariableOp"lstm_cell_75/MatMul/ReadVariableOp2L
$lstm_cell_75/MatMul_1/ReadVariableOp$lstm_cell_75/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
[
¯
#__inference__traced_restore_3248063
file_prefix3
 assignvariableop_dense_72_kernel:	
.
 assignvariableop_1_dense_72_bias:
A
.assignvariableop_2_lstm_75_lstm_cell_75_kernel:	L
8assignvariableop_3_lstm_75_lstm_cell_75_recurrent_kernel:
;
,assignvariableop_4_lstm_75_lstm_cell_75_bias:	&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: #
assignvariableop_10_total: #
assignvariableop_11_count: =
*assignvariableop_12_adam_dense_72_kernel_m:	
6
(assignvariableop_13_adam_dense_72_bias_m:
I
6assignvariableop_14_adam_lstm_75_lstm_cell_75_kernel_m:	T
@assignvariableop_15_adam_lstm_75_lstm_cell_75_recurrent_kernel_m:
C
4assignvariableop_16_adam_lstm_75_lstm_cell_75_bias_m:	=
*assignvariableop_17_adam_dense_72_kernel_v:	
6
(assignvariableop_18_adam_dense_72_bias_v:
I
6assignvariableop_19_adam_lstm_75_lstm_cell_75_kernel_v:	T
@assignvariableop_20_adam_lstm_75_lstm_cell_75_recurrent_kernel_v:
C
4assignvariableop_21_adam_lstm_75_lstm_cell_75_bias_v:	
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
AssignVariableOpAssignVariableOp assignvariableop_dense_72_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_72_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp.assignvariableop_2_lstm_75_lstm_cell_75_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_3AssignVariableOp8assignvariableop_3_lstm_75_lstm_cell_75_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_75_lstm_cell_75_biasIdentity_4:output:0"/device:CPU:0*
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
AssignVariableOp_12AssignVariableOp*assignvariableop_12_adam_dense_72_kernel_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp(assignvariableop_13_adam_dense_72_bias_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_14AssignVariableOp6assignvariableop_14_adam_lstm_75_lstm_cell_75_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_15AssignVariableOp@assignvariableop_15_adam_lstm_75_lstm_cell_75_recurrent_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_16AssignVariableOp4assignvariableop_16_adam_lstm_75_lstm_cell_75_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_72_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_72_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_lstm_75_lstm_cell_75_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_20AssignVariableOp@assignvariableop_20_adam_lstm_75_lstm_cell_75_recurrent_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_lstm_75_lstm_cell_75_bias_vIdentity_21:output:0"/device:CPU:0*
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
$
ì
while_body_3246155
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_75_3246179_0:	0
while_lstm_cell_75_3246181_0:
+
while_lstm_cell_75_3246183_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_75_3246179:	.
while_lstm_cell_75_3246181:
)
while_lstm_cell_75_3246183:	¢*while/lstm_cell_75/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0º
*while/lstm_cell_75/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_75_3246179_0while_lstm_cell_75_3246181_0while_lstm_cell_75_3246183_0*
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
I__inference_lstm_cell_75_layer_call_and_return_conditional_losses_3246095r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_75/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_75/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity3while/lstm_cell_75/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

while/NoOpNoOp+^while/lstm_cell_75/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_75_3246179while_lstm_cell_75_3246179_0":
while_lstm_cell_75_3246181while_lstm_cell_75_3246181_0":
while_lstm_cell_75_3246183while_lstm_cell_75_3246183_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_75/StatefulPartitionedCall*while/lstm_cell_75/StatefulPartitionedCall: 
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
$
ì
while_body_3245962
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_75_3245986_0:	0
while_lstm_cell_75_3245988_0:
+
while_lstm_cell_75_3245990_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_75_3245986:	.
while_lstm_cell_75_3245988:
)
while_lstm_cell_75_3245990:	¢*while/lstm_cell_75/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0º
*while/lstm_cell_75/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_75_3245986_0while_lstm_cell_75_3245988_0while_lstm_cell_75_3245990_0*
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
I__inference_lstm_cell_75_layer_call_and_return_conditional_losses_3245947r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_75/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_75/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity3while/lstm_cell_75/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

while/NoOpNoOp+^while/lstm_cell_75/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_75_3245986while_lstm_cell_75_3245986_0":
while_lstm_cell_75_3245988while_lstm_cell_75_3245988_0":
while_lstm_cell_75_3245990while_lstm_cell_75_3245990_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_75/StatefulPartitionedCall*while/lstm_cell_75/StatefulPartitionedCall: 
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
¾
È
while_cond_3247654
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3247654___redundant_placeholder05
1while_while_cond_3247654___redundant_placeholder15
1while_while_cond_3247654___redundant_placeholder25
1while_while_cond_3247654___redundant_placeholder3
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
while_cond_3246154
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3246154___redundant_placeholder05
1while_while_cond_3246154___redundant_placeholder15
1while_while_cond_3246154___redundant_placeholder25
1while_while_cond_3246154___redundant_placeholder3
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
while_body_3246300
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_75_matmul_readvariableop_resource_0:	I
5while_lstm_cell_75_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_75_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_75_matmul_readvariableop_resource:	G
3while_lstm_cell_75_matmul_1_readvariableop_resource:
A
2while_lstm_cell_75_biasadd_readvariableop_resource:	¢)while/lstm_cell_75/BiasAdd/ReadVariableOp¢(while/lstm_cell_75/MatMul/ReadVariableOp¢*while/lstm_cell_75/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_75/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_75_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_75/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_75/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_75_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_75/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_75/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/addAddV2#while/lstm_cell_75/MatMul:product:0%while/lstm_cell_75/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_75/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_75_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_75/BiasAddBiasAddwhile/lstm_cell_75/add:z:01while/lstm_cell_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_75/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_75/splitSplit+while/lstm_cell_75/split/split_dim:output:0#while/lstm_cell_75/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_75/SigmoidSigmoid!while/lstm_cell_75/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_75/Sigmoid_1Sigmoid!while/lstm_cell_75/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/mulMul while/lstm_cell_75/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_75/ReluRelu!while/lstm_cell_75/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/mul_1Mulwhile/lstm_cell_75/Sigmoid:y:0%while/lstm_cell_75/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/add_1AddV2while/lstm_cell_75/mul:z:0while/lstm_cell_75/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_75/Sigmoid_2Sigmoid!while/lstm_cell_75/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_75/Relu_1Reluwhile/lstm_cell_75/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_75/mul_2Mul while/lstm_cell_75/Sigmoid_2:y:0'while/lstm_cell_75/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_75/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_75/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_75/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_75/BiasAdd/ReadVariableOp)^while/lstm_cell_75/MatMul/ReadVariableOp+^while/lstm_cell_75/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_75_biasadd_readvariableop_resource4while_lstm_cell_75_biasadd_readvariableop_resource_0"l
3while_lstm_cell_75_matmul_1_readvariableop_resource5while_lstm_cell_75_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_75_matmul_readvariableop_resource3while_lstm_cell_75_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_75/BiasAdd/ReadVariableOp)while/lstm_cell_75/BiasAdd/ReadVariableOp2T
(while/lstm_cell_75/MatMul/ReadVariableOp(while/lstm_cell_75/MatMul/ReadVariableOp2X
*while/lstm_cell_75/MatMul_1/ReadVariableOp*while/lstm_cell_75/MatMul_1/ReadVariableOp: 
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
+__inference_dropout_8_layer_call_fn_3247750

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
F__inference_dropout_8_layer_call_and_return_conditional_losses_3246477p
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
¥
G
+__inference_dropout_8_layer_call_fn_3247745

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
F__inference_dropout_8_layer_call_and_return_conditional_losses_3246398a
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
input_485
serving_default_input_48:0ÿÿÿÿÿÿÿÿÿ
I
weighted_layer_334
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
(__inference_ma_mod_layer_call_fn_3246440
(__inference_ma_mod_layer_call_fn_3246784
(__inference_ma_mod_layer_call_fn_3246799
(__inference_ma_mod_layer_call_fn_3246710À
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
C__inference_ma_mod_layer_call_and_return_conditional_losses_3246954
C__inference_ma_mod_layer_call_and_return_conditional_losses_3247116
C__inference_ma_mod_layer_call_and_return_conditional_losses_3246728
C__inference_ma_mod_layer_call_and_return_conditional_losses_3246746À
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
"__inference__wrapped_model_3245880input_48"
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
)__inference_lstm_75_layer_call_fn_3247127
)__inference_lstm_75_layer_call_fn_3247138
)__inference_lstm_75_layer_call_fn_3247149
)__inference_lstm_75_layer_call_fn_3247160Õ
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
D__inference_lstm_75_layer_call_and_return_conditional_losses_3247305
D__inference_lstm_75_layer_call_and_return_conditional_losses_3247450
D__inference_lstm_75_layer_call_and_return_conditional_losses_3247595
D__inference_lstm_75_layer_call_and_return_conditional_losses_3247740Õ
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
+__inference_dropout_8_layer_call_fn_3247745
+__inference_dropout_8_layer_call_fn_3247750´
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
F__inference_dropout_8_layer_call_and_return_conditional_losses_3247755
F__inference_dropout_8_layer_call_and_return_conditional_losses_3247767´
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
*__inference_dense_72_layer_call_fn_3247776¢
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
E__inference_dense_72_layer_call_and_return_conditional_losses_3247786¢
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
2dense_72/kernel
:
2dense_72/bias
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
3__inference_weighted_layer_33_layer_call_fn_3247792¢
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
N__inference_weighted_layer_33_layer_call_and_return_conditional_losses_3247800¢
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
.:,	2lstm_75/lstm_cell_75/kernel
9:7
2%lstm_75/lstm_cell_75/recurrent_kernel
(:&2lstm_75/lstm_cell_75/bias
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
(__inference_ma_mod_layer_call_fn_3246440input_48"À
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
(__inference_ma_mod_layer_call_fn_3246784inputs"À
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
(__inference_ma_mod_layer_call_fn_3246799inputs"À
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
(__inference_ma_mod_layer_call_fn_3246710input_48"À
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
C__inference_ma_mod_layer_call_and_return_conditional_losses_3246954inputs"À
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
C__inference_ma_mod_layer_call_and_return_conditional_losses_3247116inputs"À
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
C__inference_ma_mod_layer_call_and_return_conditional_losses_3246728input_48"À
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
C__inference_ma_mod_layer_call_and_return_conditional_losses_3246746input_48"À
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
%__inference_signature_wrapper_3246769input_48"
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
)__inference_lstm_75_layer_call_fn_3247127inputs/0"Õ
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
)__inference_lstm_75_layer_call_fn_3247138inputs/0"Õ
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
)__inference_lstm_75_layer_call_fn_3247149inputs"Õ
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
)__inference_lstm_75_layer_call_fn_3247160inputs"Õ
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
D__inference_lstm_75_layer_call_and_return_conditional_losses_3247305inputs/0"Õ
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
D__inference_lstm_75_layer_call_and_return_conditional_losses_3247450inputs/0"Õ
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
D__inference_lstm_75_layer_call_and_return_conditional_losses_3247595inputs"Õ
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
D__inference_lstm_75_layer_call_and_return_conditional_losses_3247740inputs"Õ
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
.__inference_lstm_cell_75_layer_call_fn_3247817
.__inference_lstm_cell_75_layer_call_fn_3247834¾
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
I__inference_lstm_cell_75_layer_call_and_return_conditional_losses_3247866
I__inference_lstm_cell_75_layer_call_and_return_conditional_losses_3247898¾
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
+__inference_dropout_8_layer_call_fn_3247745inputs"´
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
+__inference_dropout_8_layer_call_fn_3247750inputs"´
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
F__inference_dropout_8_layer_call_and_return_conditional_losses_3247755inputs"´
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
F__inference_dropout_8_layer_call_and_return_conditional_losses_3247767inputs"´
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
*__inference_dense_72_layer_call_fn_3247776inputs"¢
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
E__inference_dense_72_layer_call_and_return_conditional_losses_3247786inputs"¢
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
3__inference_weighted_layer_33_layer_call_fn_3247792inputs/0inputs/1"¢
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
N__inference_weighted_layer_33_layer_call_and_return_conditional_losses_3247800inputs/0inputs/1"¢
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
.__inference_lstm_cell_75_layer_call_fn_3247817inputsstates/0states/1"¾
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
.__inference_lstm_cell_75_layer_call_fn_3247834inputsstates/0states/1"¾
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
I__inference_lstm_cell_75_layer_call_and_return_conditional_losses_3247866inputsstates/0states/1"¾
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
I__inference_lstm_cell_75_layer_call_and_return_conditional_losses_3247898inputsstates/0states/1"¾
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
2Adam/dense_72/kernel/m
 :
2Adam/dense_72/bias/m
3:1	2"Adam/lstm_75/lstm_cell_75/kernel/m
>:<
2,Adam/lstm_75/lstm_cell_75/recurrent_kernel/m
-:+2 Adam/lstm_75/lstm_cell_75/bias/m
':%	
2Adam/dense_72/kernel/v
 :
2Adam/dense_72/bias/v
3:1	2"Adam/lstm_75/lstm_cell_75/kernel/v
>:<
2,Adam/lstm_75/lstm_cell_75/recurrent_kernel/v
-:+2 Adam/lstm_75/lstm_cell_75/bias/v°
"__inference__wrapped_model_3245880-./%&5¢2
+¢(
&#
input_48ÿÿÿÿÿÿÿÿÿ

ª "IªF
D
weighted_layer_33/,
weighted_layer_33ÿÿÿÿÿÿÿÿÿ
¦
E__inference_dense_72_layer_call_and_return_conditional_losses_3247786]%&0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ~
*__inference_dense_72_layer_call_fn_3247776P%&0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
¨
F__inference_dropout_8_layer_call_and_return_conditional_losses_3247755^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¨
F__inference_dropout_8_layer_call_and_return_conditional_losses_3247767^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dropout_8_layer_call_fn_3247745Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_dropout_8_layer_call_fn_3247750Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÆ
D__inference_lstm_75_layer_call_and_return_conditional_losses_3247305~-./O¢L
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
D__inference_lstm_75_layer_call_and_return_conditional_losses_3247450~-./O¢L
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
D__inference_lstm_75_layer_call_and_return_conditional_losses_3247595n-./?¢<
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
D__inference_lstm_75_layer_call_and_return_conditional_losses_3247740n-./?¢<
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
)__inference_lstm_75_layer_call_fn_3247127q-./O¢L
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
)__inference_lstm_75_layer_call_fn_3247138q-./O¢L
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
)__inference_lstm_75_layer_call_fn_3247149a-./?¢<
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
)__inference_lstm_75_layer_call_fn_3247160a-./?¢<
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
I__inference_lstm_cell_75_layer_call_and_return_conditional_losses_3247866-./¢
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
I__inference_lstm_cell_75_layer_call_and_return_conditional_losses_3247898-./¢
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
.__inference_lstm_cell_75_layer_call_fn_3247817ò-./¢
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
.__inference_lstm_cell_75_layer_call_fn_3247834ò-./¢
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
1/1ÿÿÿÿÿÿÿÿÿ¸
C__inference_ma_mod_layer_call_and_return_conditional_losses_3246728q-./%&=¢:
3¢0
&#
input_48ÿÿÿÿÿÿÿÿÿ

p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ

 ¸
C__inference_ma_mod_layer_call_and_return_conditional_losses_3246746q-./%&=¢:
3¢0
&#
input_48ÿÿÿÿÿÿÿÿÿ

p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ

 ¶
C__inference_ma_mod_layer_call_and_return_conditional_losses_3246954o-./%&;¢8
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
C__inference_ma_mod_layer_call_and_return_conditional_losses_3247116o-./%&;¢8
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
(__inference_ma_mod_layer_call_fn_3246440d-./%&=¢:
3¢0
&#
input_48ÿÿÿÿÿÿÿÿÿ

p 

 
ª "ÿÿÿÿÿÿÿÿÿ

(__inference_ma_mod_layer_call_fn_3246710d-./%&=¢:
3¢0
&#
input_48ÿÿÿÿÿÿÿÿÿ

p

 
ª "ÿÿÿÿÿÿÿÿÿ

(__inference_ma_mod_layer_call_fn_3246784b-./%&;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ

p 

 
ª "ÿÿÿÿÿÿÿÿÿ

(__inference_ma_mod_layer_call_fn_3246799b-./%&;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ

p

 
ª "ÿÿÿÿÿÿÿÿÿ
¿
%__inference_signature_wrapper_3246769-./%&A¢>
¢ 
7ª4
2
input_48&#
input_48ÿÿÿÿÿÿÿÿÿ
"IªF
D
weighted_layer_33/,
weighted_layer_33ÿÿÿÿÿÿÿÿÿ
Þ
N__inference_weighted_layer_33_layer_call_and_return_conditional_losses_3247800^¢[
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
3__inference_weighted_layer_33_layer_call_fn_3247792~^¢[
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