��0
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
dtypetype�
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
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
�
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��*
�
residual_8/conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameresidual_8/conv2d_20/bias
�
-residual_8/conv2d_20/bias/Read/ReadVariableOpReadVariableOpresidual_8/conv2d_20/bias*
_output_shapes	
:�*
dtype0
�
residual_8/conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*,
shared_nameresidual_8/conv2d_20/kernel
�
/residual_8/conv2d_20/kernel/Read/ReadVariableOpReadVariableOpresidual_8/conv2d_20/kernel*(
_output_shapes
:��*
dtype0
�
residual_8/conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameresidual_8/conv2d_19/bias
�
-residual_8/conv2d_19/bias/Read/ReadVariableOpReadVariableOpresidual_8/conv2d_19/bias*
_output_shapes	
:�*
dtype0
�
residual_8/conv2d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*,
shared_nameresidual_8/conv2d_19/kernel
�
/residual_8/conv2d_19/kernel/Read/ReadVariableOpReadVariableOpresidual_8/conv2d_19/kernel*(
_output_shapes
:��*
dtype0
�
residual_7/conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameresidual_7/conv2d_18/bias
�
-residual_7/conv2d_18/bias/Read/ReadVariableOpReadVariableOpresidual_7/conv2d_18/bias*
_output_shapes	
:�*
dtype0
�
residual_7/conv2d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*,
shared_nameresidual_7/conv2d_18/kernel
�
/residual_7/conv2d_18/kernel/Read/ReadVariableOpReadVariableOpresidual_7/conv2d_18/kernel*(
_output_shapes
:��*
dtype0
�
residual_7/conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameresidual_7/conv2d_17/bias
�
-residual_7/conv2d_17/bias/Read/ReadVariableOpReadVariableOpresidual_7/conv2d_17/bias*
_output_shapes	
:�*
dtype0
�
residual_7/conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*,
shared_nameresidual_7/conv2d_17/kernel
�
/residual_7/conv2d_17/kernel/Read/ReadVariableOpReadVariableOpresidual_7/conv2d_17/kernel*(
_output_shapes
:��*
dtype0
�
residual_6/conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameresidual_6/conv2d_16/bias
�
-residual_6/conv2d_16/bias/Read/ReadVariableOpReadVariableOpresidual_6/conv2d_16/bias*
_output_shapes	
:�*
dtype0
�
residual_6/conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*,
shared_nameresidual_6/conv2d_16/kernel
�
/residual_6/conv2d_16/kernel/Read/ReadVariableOpReadVariableOpresidual_6/conv2d_16/kernel*(
_output_shapes
:��*
dtype0
�
residual_6/conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameresidual_6/conv2d_15/bias
�
-residual_6/conv2d_15/bias/Read/ReadVariableOpReadVariableOpresidual_6/conv2d_15/bias*
_output_shapes	
:�*
dtype0
�
residual_6/conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*,
shared_nameresidual_6/conv2d_15/kernel
�
/residual_6/conv2d_15/kernel/Read/ReadVariableOpReadVariableOpresidual_6/conv2d_15/kernel*(
_output_shapes
:��*
dtype0
�
residual_5/conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameresidual_5/conv2d_14/bias
�
-residual_5/conv2d_14/bias/Read/ReadVariableOpReadVariableOpresidual_5/conv2d_14/bias*
_output_shapes	
:�*
dtype0
�
residual_5/conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*,
shared_nameresidual_5/conv2d_14/kernel
�
/residual_5/conv2d_14/kernel/Read/ReadVariableOpReadVariableOpresidual_5/conv2d_14/kernel*(
_output_shapes
:��*
dtype0
�
residual_5/conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameresidual_5/conv2d_13/bias
�
-residual_5/conv2d_13/bias/Read/ReadVariableOpReadVariableOpresidual_5/conv2d_13/bias*
_output_shapes	
:�*
dtype0
�
residual_5/conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*,
shared_nameresidual_5/conv2d_13/kernel
�
/residual_5/conv2d_13/kernel/Read/ReadVariableOpReadVariableOpresidual_5/conv2d_13/kernel*(
_output_shapes
:��*
dtype0
�
residual_4/conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameresidual_4/conv2d_12/bias
�
-residual_4/conv2d_12/bias/Read/ReadVariableOpReadVariableOpresidual_4/conv2d_12/bias*
_output_shapes	
:�*
dtype0
�
residual_4/conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*,
shared_nameresidual_4/conv2d_12/kernel
�
/residual_4/conv2d_12/kernel/Read/ReadVariableOpReadVariableOpresidual_4/conv2d_12/kernel*(
_output_shapes
:��*
dtype0
�
residual_4/conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameresidual_4/conv2d_11/bias
�
-residual_4/conv2d_11/bias/Read/ReadVariableOpReadVariableOpresidual_4/conv2d_11/bias*
_output_shapes	
:�*
dtype0
�
residual_4/conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*,
shared_nameresidual_4/conv2d_11/kernel
�
/residual_4/conv2d_11/kernel/Read/ReadVariableOpReadVariableOpresidual_4/conv2d_11/kernel*(
_output_shapes
:��*
dtype0
�
residual_3/conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameresidual_3/conv2d_10/bias
�
-residual_3/conv2d_10/bias/Read/ReadVariableOpReadVariableOpresidual_3/conv2d_10/bias*
_output_shapes	
:�*
dtype0
�
residual_3/conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*,
shared_nameresidual_3/conv2d_10/kernel
�
/residual_3/conv2d_10/kernel/Read/ReadVariableOpReadVariableOpresidual_3/conv2d_10/kernel*(
_output_shapes
:��*
dtype0
�
residual_3/conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameresidual_3/conv2d_9/bias
�
,residual_3/conv2d_9/bias/Read/ReadVariableOpReadVariableOpresidual_3/conv2d_9/bias*
_output_shapes	
:�*
dtype0
�
residual_3/conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*+
shared_nameresidual_3/conv2d_9/kernel
�
.residual_3/conv2d_9/kernel/Read/ReadVariableOpReadVariableOpresidual_3/conv2d_9/kernel*(
_output_shapes
:��*
dtype0
�
residual_2/conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameresidual_2/conv2d_8/bias
�
,residual_2/conv2d_8/bias/Read/ReadVariableOpReadVariableOpresidual_2/conv2d_8/bias*
_output_shapes	
:�*
dtype0
�
residual_2/conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*+
shared_nameresidual_2/conv2d_8/kernel
�
.residual_2/conv2d_8/kernel/Read/ReadVariableOpReadVariableOpresidual_2/conv2d_8/kernel*(
_output_shapes
:��*
dtype0
�
residual_2/conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameresidual_2/conv2d_7/bias
�
,residual_2/conv2d_7/bias/Read/ReadVariableOpReadVariableOpresidual_2/conv2d_7/bias*
_output_shapes	
:�*
dtype0
�
residual_2/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*+
shared_nameresidual_2/conv2d_7/kernel
�
.residual_2/conv2d_7/kernel/Read/ReadVariableOpReadVariableOpresidual_2/conv2d_7/kernel*(
_output_shapes
:��*
dtype0
�
residual_1/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameresidual_1/conv2d_6/bias
�
,residual_1/conv2d_6/bias/Read/ReadVariableOpReadVariableOpresidual_1/conv2d_6/bias*
_output_shapes	
:�*
dtype0
�
residual_1/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*+
shared_nameresidual_1/conv2d_6/kernel
�
.residual_1/conv2d_6/kernel/Read/ReadVariableOpReadVariableOpresidual_1/conv2d_6/kernel*(
_output_shapes
:��*
dtype0
�
residual_1/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameresidual_1/conv2d_5/bias
�
,residual_1/conv2d_5/bias/Read/ReadVariableOpReadVariableOpresidual_1/conv2d_5/bias*
_output_shapes	
:�*
dtype0
�
residual_1/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*+
shared_nameresidual_1/conv2d_5/kernel
�
.residual_1/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpresidual_1/conv2d_5/kernel*(
_output_shapes
:��*
dtype0
�
residual/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameresidual/conv2d_4/bias
~
*residual/conv2d_4/bias/Read/ReadVariableOpReadVariableOpresidual/conv2d_4/bias*
_output_shapes	
:�*
dtype0
�
residual/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*)
shared_nameresidual/conv2d_4/kernel
�
,residual/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpresidual/conv2d_4/kernel*(
_output_shapes
:��*
dtype0
�
residual/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameresidual/conv2d_3/bias
~
*residual/conv2d_3/bias/Read/ReadVariableOpReadVariableOpresidual/conv2d_3/bias*
_output_shapes	
:�*
dtype0
�
residual/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*)
shared_nameresidual/conv2d_3/kernel
�
,residual/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpresidual/conv2d_3/kernel*(
_output_shapes
:��*
dtype0
t
conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_21/bias
m
"conv2d_21/bias/Read/ReadVariableOpReadVariableOpconv2d_21/bias*
_output_shapes
:*
dtype0
�
conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_21/kernel
}
$conv2d_21/kernel/Read/ReadVariableOpReadVariableOpconv2d_21/kernel*&
_output_shapes
:@*
dtype0
�
instance_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameinstance_normalization_4/beta
�
1instance_normalization_4/beta/Read/ReadVariableOpReadVariableOpinstance_normalization_4/beta*
_output_shapes
:@*
dtype0
�
instance_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name instance_normalization_4/gamma
�
2instance_normalization_4/gamma/Read/ReadVariableOpReadVariableOpinstance_normalization_4/gamma*
_output_shapes
:@*
dtype0
�
conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameconv2d_transpose_1/bias

+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes
:@*
dtype0
�
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�**
shared_nameconv2d_transpose_1/kernel
�
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*'
_output_shapes
:@�*
dtype0
�
instance_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_nameinstance_normalization_3/beta
�
1instance_normalization_3/beta/Read/ReadVariableOpReadVariableOpinstance_normalization_3/beta*
_output_shapes	
:�*
dtype0
�
instance_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*/
shared_name instance_normalization_3/gamma
�
2instance_normalization_3/gamma/Read/ReadVariableOpReadVariableOpinstance_normalization_3/gamma*
_output_shapes	
:�*
dtype0
�
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameconv2d_transpose/bias
|
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes	
:�*
dtype0
�
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameconv2d_transpose/kernel
�
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*(
_output_shapes
:��*
dtype0
�
instance_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_nameinstance_normalization_2/beta
�
1instance_normalization_2/beta/Read/ReadVariableOpReadVariableOpinstance_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
instance_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*/
shared_name instance_normalization_2/gamma
�
2instance_normalization_2/gamma/Read/ReadVariableOpReadVariableOpinstance_normalization_2/gamma*
_output_shapes	
:�*
dtype0
s
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_2/bias
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:�*
dtype0
�
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameconv2d_2/kernel
}
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*(
_output_shapes
:��*
dtype0
�
instance_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_nameinstance_normalization_1/beta
�
1instance_normalization_1/beta/Read/ReadVariableOpReadVariableOpinstance_normalization_1/beta*
_output_shapes	
:�*
dtype0
�
instance_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*/
shared_name instance_normalization_1/gamma
�
2instance_normalization_1/gamma/Read/ReadVariableOpReadVariableOpinstance_normalization_1/gamma*
_output_shapes	
:�*
dtype0
s
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_1/bias
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes	
:�*
dtype0
�
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�* 
shared_nameconv2d_1/kernel
|
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*'
_output_shapes
:@�*
dtype0
�
instance_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameinstance_normalization/beta
�
/instance_normalization/beta/Read/ReadVariableOpReadVariableOpinstance_normalization/beta*
_output_shapes
:@*
dtype0
�
instance_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameinstance_normalization/gamma
�
0instance_normalization/gamma/Read/ReadVariableOpReadVariableOpinstance_normalization/gamma*
_output_shapes
:@*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:@*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@*
dtype0
�
serving_default_input_1Placeholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasinstance_normalization/gammainstance_normalization/betaconv2d_1/kernelconv2d_1/biasinstance_normalization_1/gammainstance_normalization_1/betaconv2d_2/kernelconv2d_2/biasinstance_normalization_2/gammainstance_normalization_2/betaresidual/conv2d_3/kernelresidual/conv2d_3/biasresidual/conv2d_4/kernelresidual/conv2d_4/biasresidual_1/conv2d_5/kernelresidual_1/conv2d_5/biasresidual_1/conv2d_6/kernelresidual_1/conv2d_6/biasresidual_2/conv2d_7/kernelresidual_2/conv2d_7/biasresidual_2/conv2d_8/kernelresidual_2/conv2d_8/biasresidual_3/conv2d_9/kernelresidual_3/conv2d_9/biasresidual_3/conv2d_10/kernelresidual_3/conv2d_10/biasresidual_4/conv2d_11/kernelresidual_4/conv2d_11/biasresidual_4/conv2d_12/kernelresidual_4/conv2d_12/biasresidual_5/conv2d_13/kernelresidual_5/conv2d_13/biasresidual_5/conv2d_14/kernelresidual_5/conv2d_14/biasresidual_6/conv2d_15/kernelresidual_6/conv2d_15/biasresidual_6/conv2d_16/kernelresidual_6/conv2d_16/biasresidual_7/conv2d_17/kernelresidual_7/conv2d_17/biasresidual_7/conv2d_18/kernelresidual_7/conv2d_18/biasresidual_8/conv2d_19/kernelresidual_8/conv2d_19/biasresidual_8/conv2d_20/kernelresidual_8/conv2d_20/biasconv2d_transpose/kernelconv2d_transpose/biasinstance_normalization_3/gammainstance_normalization_3/betaconv2d_transpose_1/kernelconv2d_transpose_1/biasinstance_normalization_4/gammainstance_normalization_4/betaconv2d_21/kernelconv2d_21/bias*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *-
f(R&
$__inference_signature_wrapper_743035

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ڊ
valueϊBˊ BÊ
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer_with_weights-10
layer-13
layer_with_weights-11
layer-14
layer_with_weights-12
layer-15
layer_with_weights-13
layer-16
layer_with_weights-14
layer-17
layer_with_weights-15
layer-18
layer_with_weights-16
layer-19
layer-20
layer_with_weights-17
layer-21
layer_with_weights-18
layer-22
layer-23
layer_with_weights-19
layer-24
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _default_save_signature
!
signatures*
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias
 *_jit_compiled_convolution_op*
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
	1gamma
2beta*
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses* 
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias
 A_jit_compiled_convolution_op*
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
	Hgamma
Ibeta*
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses* 
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

Vkernel
Wbias
 X_jit_compiled_convolution_op*
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
	_gamma
`beta*
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses* 
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses
	mconv1
	nconv2*
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses
	uconv1
	vconv2*
�
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses
	}conv1
	~conv2*
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�conv1

�conv2*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�conv1

�conv2*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�conv1

�conv2*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�conv1

�conv2*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�conv1

�conv2*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�conv1

�conv2*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�gamma
	�beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�gamma
	�beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
(0
)1
12
23
?4
@5
H6
I7
V8
W9
_10
`11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57*
�
(0
)1
12
23
?4
@5
H6
I7
V8
W9
_10
`11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
 _default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 

�serving_default* 

(0
)1*

(0
)1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

10
21*

10
21*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ke
VARIABLE_VALUEinstance_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEinstance_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

?0
@1*

?0
@1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

H0
I1*

H0
I1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
mg
VARIABLE_VALUEinstance_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEinstance_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

V0
W1*

V0
W1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

_0
`1*

_0
`1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
mg
VARIABLE_VALUEinstance_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEinstance_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
$
�0
�1
�2
�3*
$
�0
�1
�2
�3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
$
�0
�1
�2
�3*
$
�0
�1
�2
�3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
$
�0
�1
�2
�3*
$
�0
�1
�2
�3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
$
�0
�1
�2
�3*
$
�0
�1
�2
�3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
$
�0
�1
�2
�3*
$
�0
�1
�2
�3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
$
�0
�1
�2
�3*
$
�0
�1
�2
�3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
$
�0
�1
�2
�3*
$
�0
�1
�2
�3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
$
�0
�1
�2
�3*
$
�0
�1
�2
�3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
$
�0
�1
�2
�3*
$
�0
�1
�2
�3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
hb
VARIABLE_VALUEconv2d_transpose/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEconv2d_transpose/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
nh
VARIABLE_VALUEinstance_normalization_3/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEinstance_normalization_3/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
jd
VARIABLE_VALUEconv2d_transpose_1/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_1/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
nh
VARIABLE_VALUEinstance_normalization_4/gamma6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEinstance_normalization_4/beta5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEconv2d_21/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_21/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
YS
VARIABLE_VALUEresidual/conv2d_3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEresidual/conv2d_3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEresidual/conv2d_4/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEresidual/conv2d_4/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEresidual_1/conv2d_5/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEresidual_1/conv2d_5/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEresidual_1/conv2d_6/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEresidual_1/conv2d_6/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEresidual_2/conv2d_7/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEresidual_2/conv2d_7/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEresidual_2/conv2d_8/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEresidual_2/conv2d_8/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEresidual_3/conv2d_9/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEresidual_3/conv2d_9/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEresidual_3/conv2d_10/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEresidual_3/conv2d_10/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEresidual_4/conv2d_11/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEresidual_4/conv2d_11/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEresidual_4/conv2d_12/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEresidual_4/conv2d_12/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEresidual_5/conv2d_13/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEresidual_5/conv2d_13/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEresidual_5/conv2d_14/kernel'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEresidual_5/conv2d_14/bias'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEresidual_6/conv2d_15/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEresidual_6/conv2d_15/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEresidual_6/conv2d_16/kernel'variables/38/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEresidual_6/conv2d_16/bias'variables/39/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEresidual_7/conv2d_17/kernel'variables/40/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEresidual_7/conv2d_17/bias'variables/41/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEresidual_7/conv2d_18/kernel'variables/42/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEresidual_7/conv2d_18/bias'variables/43/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEresidual_8/conv2d_19/kernel'variables/44/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEresidual_8/conv2d_19/bias'variables/45/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEresidual_8/conv2d_20/kernel'variables/46/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEresidual_8/conv2d_20/bias'variables/47/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24*
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
* 

m0
n1*
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

u0
v1*
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

}0
~1*
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
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
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasinstance_normalization/gammainstance_normalization/betaconv2d_1/kernelconv2d_1/biasinstance_normalization_1/gammainstance_normalization_1/betaconv2d_2/kernelconv2d_2/biasinstance_normalization_2/gammainstance_normalization_2/betaconv2d_transpose/kernelconv2d_transpose/biasinstance_normalization_3/gammainstance_normalization_3/betaconv2d_transpose_1/kernelconv2d_transpose_1/biasinstance_normalization_4/gammainstance_normalization_4/betaconv2d_21/kernelconv2d_21/biasresidual/conv2d_3/kernelresidual/conv2d_3/biasresidual/conv2d_4/kernelresidual/conv2d_4/biasresidual_1/conv2d_5/kernelresidual_1/conv2d_5/biasresidual_1/conv2d_6/kernelresidual_1/conv2d_6/biasresidual_2/conv2d_7/kernelresidual_2/conv2d_7/biasresidual_2/conv2d_8/kernelresidual_2/conv2d_8/biasresidual_3/conv2d_9/kernelresidual_3/conv2d_9/biasresidual_3/conv2d_10/kernelresidual_3/conv2d_10/biasresidual_4/conv2d_11/kernelresidual_4/conv2d_11/biasresidual_4/conv2d_12/kernelresidual_4/conv2d_12/biasresidual_5/conv2d_13/kernelresidual_5/conv2d_13/biasresidual_5/conv2d_14/kernelresidual_5/conv2d_14/biasresidual_6/conv2d_15/kernelresidual_6/conv2d_15/biasresidual_6/conv2d_16/kernelresidual_6/conv2d_16/biasresidual_7/conv2d_17/kernelresidual_7/conv2d_17/biasresidual_7/conv2d_18/kernelresidual_7/conv2d_18/biasresidual_8/conv2d_19/kernelresidual_8/conv2d_19/biasresidual_8/conv2d_20/kernelresidual_8/conv2d_20/biasConst*G
Tin@
>2<*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *(
f#R!
__inference__traced_save_745562
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasinstance_normalization/gammainstance_normalization/betaconv2d_1/kernelconv2d_1/biasinstance_normalization_1/gammainstance_normalization_1/betaconv2d_2/kernelconv2d_2/biasinstance_normalization_2/gammainstance_normalization_2/betaconv2d_transpose/kernelconv2d_transpose/biasinstance_normalization_3/gammainstance_normalization_3/betaconv2d_transpose_1/kernelconv2d_transpose_1/biasinstance_normalization_4/gammainstance_normalization_4/betaconv2d_21/kernelconv2d_21/biasresidual/conv2d_3/kernelresidual/conv2d_3/biasresidual/conv2d_4/kernelresidual/conv2d_4/biasresidual_1/conv2d_5/kernelresidual_1/conv2d_5/biasresidual_1/conv2d_6/kernelresidual_1/conv2d_6/biasresidual_2/conv2d_7/kernelresidual_2/conv2d_7/biasresidual_2/conv2d_8/kernelresidual_2/conv2d_8/biasresidual_3/conv2d_9/kernelresidual_3/conv2d_9/biasresidual_3/conv2d_10/kernelresidual_3/conv2d_10/biasresidual_4/conv2d_11/kernelresidual_4/conv2d_11/biasresidual_4/conv2d_12/kernelresidual_4/conv2d_12/biasresidual_5/conv2d_13/kernelresidual_5/conv2d_13/biasresidual_5/conv2d_14/kernelresidual_5/conv2d_14/biasresidual_6/conv2d_15/kernelresidual_6/conv2d_15/biasresidual_6/conv2d_16/kernelresidual_6/conv2d_16/biasresidual_7/conv2d_17/kernelresidual_7/conv2d_17/biasresidual_7/conv2d_18/kernelresidual_7/conv2d_18/biasresidual_8/conv2d_19/kernelresidual_8/conv2d_19/biasresidual_8/conv2d_20/kernelresidual_8/conv2d_20/bias*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *+
f&R$
"__inference__traced_restore_745746��&
�
�
D__inference_conv2d_5_layer_call_and_return_conditional_losses_744899

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������@@�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
E__inference_conv2d_11_layer_call_and_return_conditional_losses_745016

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������@@�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
� 
�
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_741421

inputsC
(conv2d_transpose_readvariableop_resource:@�-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�

�
E__inference_conv2d_10_layer_call_and_return_conditional_losses_740930

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
*__inference_conv2d_14_layer_call_fn_745064

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_741078x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
)__inference_conv2d_5_layer_call_fn_744888

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_740766x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
_
C__inference_re_lu_2_layer_call_and_return_conditional_losses_741649

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:���������@@�c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:���������@@�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������@@�:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
F__inference_residual_4_layer_call_and_return_conditional_losses_744484

inputsD
(conv2d_11_conv2d_readvariableop_resource:��8
)conv2d_11_biasadd_readvariableop_resource:	�D
(conv2d_12_conv2d_readvariableop_resource:��8
)conv2d_12_biasadd_readvariableop_resource:	�
identity�� conv2d_11/BiasAdd/ReadVariableOp�conv2d_11/Conv2D/ReadVariableOp� conv2d_12/BiasAdd/ReadVariableOp�conv2d_12/Conv2D/ReadVariableOp�
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_11/Conv2DConv2Dinputs'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�m
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_12/Conv2DConv2Dconv2d_11/Relu:activations:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�o
add/addAddV2conv2d_12/BiasAdd:output:0inputs*
T0*0
_output_shapes
:���������@@�Z

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:���������@@�p
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@��
NoOpNoOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
+__inference_residual_1_layer_call_fn_744369

inputs#
unknown:��
	unknown_0:	�%
	unknown_1:��
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_1_layer_call_and_return_conditional_losses_740791x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
_
C__inference_re_lu_2_layer_call_and_return_conditional_losses_744324

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:���������@@�c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:���������@@�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������@@�:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�*
�
T__inference_instance_normalization_1_layer_call_and_return_conditional_losses_741566

inputs.
reshape_readvariableop_resource:	�0
!reshape_1_readvariableop_resource:	�
identity��Reshape/ReadVariableOp�Reshape_1/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(v
moments/StopGradientStopGradientmoments/mean:output:0*
T0*0
_output_shapes
:�����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*2
_output_shapes 
:������������s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(s
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes	
:�*
dtype0f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   |
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*'
_output_shapes
:�w
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:�T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*0
_output_shapes
:����������f
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*0
_output_shapes
:����������v
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*0
_output_shapes
:����������n
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*2
_output_shapes 
:������������{
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*0
_output_shapes
:����������x
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*0
_output_shapes
:����������}
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*2
_output_shapes 
:������������m
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*2
_output_shapes 
:������������z
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":������������: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:Z V
2
_output_shapes 
:������������
 
_user_specified_nameinputs
�
D
(__inference_re_lu_1_layer_call_fn_744238

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *L
fGRE
C__inference_re_lu_1_layer_call_and_return_conditional_losses_741577k
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������:Z V
2
_output_shapes 
:������������
 
_user_specified_nameinputs
�
�
9__inference_instance_normalization_1_layer_call_fn_744190

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *]
fXRV
T__inference_instance_normalization_1_layer_call_and_return_conditional_losses_741566z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:������������
 
_user_specified_nameinputs
��
�5
F__inference_sequential_layer_call_and_return_conditional_losses_744081

inputs?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@D
6instance_normalization_reshape_readvariableop_resource:@F
8instance_normalization_reshape_1_readvariableop_resource:@B
'conv2d_1_conv2d_readvariableop_resource:@�7
(conv2d_1_biasadd_readvariableop_resource:	�G
8instance_normalization_1_reshape_readvariableop_resource:	�I
:instance_normalization_1_reshape_1_readvariableop_resource:	�C
'conv2d_2_conv2d_readvariableop_resource:��7
(conv2d_2_biasadd_readvariableop_resource:	�G
8instance_normalization_2_reshape_readvariableop_resource:	�I
:instance_normalization_2_reshape_1_readvariableop_resource:	�L
0residual_conv2d_3_conv2d_readvariableop_resource:��@
1residual_conv2d_3_biasadd_readvariableop_resource:	�L
0residual_conv2d_4_conv2d_readvariableop_resource:��@
1residual_conv2d_4_biasadd_readvariableop_resource:	�N
2residual_1_conv2d_5_conv2d_readvariableop_resource:��B
3residual_1_conv2d_5_biasadd_readvariableop_resource:	�N
2residual_1_conv2d_6_conv2d_readvariableop_resource:��B
3residual_1_conv2d_6_biasadd_readvariableop_resource:	�N
2residual_2_conv2d_7_conv2d_readvariableop_resource:��B
3residual_2_conv2d_7_biasadd_readvariableop_resource:	�N
2residual_2_conv2d_8_conv2d_readvariableop_resource:��B
3residual_2_conv2d_8_biasadd_readvariableop_resource:	�N
2residual_3_conv2d_9_conv2d_readvariableop_resource:��B
3residual_3_conv2d_9_biasadd_readvariableop_resource:	�O
3residual_3_conv2d_10_conv2d_readvariableop_resource:��C
4residual_3_conv2d_10_biasadd_readvariableop_resource:	�O
3residual_4_conv2d_11_conv2d_readvariableop_resource:��C
4residual_4_conv2d_11_biasadd_readvariableop_resource:	�O
3residual_4_conv2d_12_conv2d_readvariableop_resource:��C
4residual_4_conv2d_12_biasadd_readvariableop_resource:	�O
3residual_5_conv2d_13_conv2d_readvariableop_resource:��C
4residual_5_conv2d_13_biasadd_readvariableop_resource:	�O
3residual_5_conv2d_14_conv2d_readvariableop_resource:��C
4residual_5_conv2d_14_biasadd_readvariableop_resource:	�O
3residual_6_conv2d_15_conv2d_readvariableop_resource:��C
4residual_6_conv2d_15_biasadd_readvariableop_resource:	�O
3residual_6_conv2d_16_conv2d_readvariableop_resource:��C
4residual_6_conv2d_16_biasadd_readvariableop_resource:	�O
3residual_7_conv2d_17_conv2d_readvariableop_resource:��C
4residual_7_conv2d_17_biasadd_readvariableop_resource:	�O
3residual_7_conv2d_18_conv2d_readvariableop_resource:��C
4residual_7_conv2d_18_biasadd_readvariableop_resource:	�O
3residual_8_conv2d_19_conv2d_readvariableop_resource:��C
4residual_8_conv2d_19_biasadd_readvariableop_resource:	�O
3residual_8_conv2d_20_conv2d_readvariableop_resource:��C
4residual_8_conv2d_20_biasadd_readvariableop_resource:	�U
9conv2d_transpose_conv2d_transpose_readvariableop_resource:��?
0conv2d_transpose_biasadd_readvariableop_resource:	�G
8instance_normalization_3_reshape_readvariableop_resource:	�I
:instance_normalization_3_reshape_1_readvariableop_resource:	�V
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@�@
2conv2d_transpose_1_biasadd_readvariableop_resource:@F
8instance_normalization_4_reshape_readvariableop_resource:@H
:instance_normalization_4_reshape_1_readvariableop_resource:@B
(conv2d_21_conv2d_readvariableop_resource:@7
)conv2d_21_biasadd_readvariableop_resource:
identity��conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp� conv2d_21/BiasAdd/ReadVariableOp�conv2d_21/Conv2D/ReadVariableOp�'conv2d_transpose/BiasAdd/ReadVariableOp�0conv2d_transpose/conv2d_transpose/ReadVariableOp�)conv2d_transpose_1/BiasAdd/ReadVariableOp�2conv2d_transpose_1/conv2d_transpose/ReadVariableOp�-instance_normalization/Reshape/ReadVariableOp�/instance_normalization/Reshape_1/ReadVariableOp�/instance_normalization_1/Reshape/ReadVariableOp�1instance_normalization_1/Reshape_1/ReadVariableOp�/instance_normalization_2/Reshape/ReadVariableOp�1instance_normalization_2/Reshape_1/ReadVariableOp�/instance_normalization_3/Reshape/ReadVariableOp�1instance_normalization_3/Reshape_1/ReadVariableOp�/instance_normalization_4/Reshape/ReadVariableOp�1instance_normalization_4/Reshape_1/ReadVariableOp�(residual/conv2d_3/BiasAdd/ReadVariableOp�'residual/conv2d_3/Conv2D/ReadVariableOp�(residual/conv2d_4/BiasAdd/ReadVariableOp�'residual/conv2d_4/Conv2D/ReadVariableOp�*residual_1/conv2d_5/BiasAdd/ReadVariableOp�)residual_1/conv2d_5/Conv2D/ReadVariableOp�*residual_1/conv2d_6/BiasAdd/ReadVariableOp�)residual_1/conv2d_6/Conv2D/ReadVariableOp�*residual_2/conv2d_7/BiasAdd/ReadVariableOp�)residual_2/conv2d_7/Conv2D/ReadVariableOp�*residual_2/conv2d_8/BiasAdd/ReadVariableOp�)residual_2/conv2d_8/Conv2D/ReadVariableOp�+residual_3/conv2d_10/BiasAdd/ReadVariableOp�*residual_3/conv2d_10/Conv2D/ReadVariableOp�*residual_3/conv2d_9/BiasAdd/ReadVariableOp�)residual_3/conv2d_9/Conv2D/ReadVariableOp�+residual_4/conv2d_11/BiasAdd/ReadVariableOp�*residual_4/conv2d_11/Conv2D/ReadVariableOp�+residual_4/conv2d_12/BiasAdd/ReadVariableOp�*residual_4/conv2d_12/Conv2D/ReadVariableOp�+residual_5/conv2d_13/BiasAdd/ReadVariableOp�*residual_5/conv2d_13/Conv2D/ReadVariableOp�+residual_5/conv2d_14/BiasAdd/ReadVariableOp�*residual_5/conv2d_14/Conv2D/ReadVariableOp�+residual_6/conv2d_15/BiasAdd/ReadVariableOp�*residual_6/conv2d_15/Conv2D/ReadVariableOp�+residual_6/conv2d_16/BiasAdd/ReadVariableOp�*residual_6/conv2d_16/Conv2D/ReadVariableOp�+residual_7/conv2d_17/BiasAdd/ReadVariableOp�*residual_7/conv2d_17/Conv2D/ReadVariableOp�+residual_7/conv2d_18/BiasAdd/ReadVariableOp�*residual_7/conv2d_18/Conv2D/ReadVariableOp�+residual_8/conv2d_19/BiasAdd/ReadVariableOp�*residual_8/conv2d_19/Conv2D/ReadVariableOp�+residual_8/conv2d_20/BiasAdd/ReadVariableOp�*residual_8/conv2d_20/Conv2D/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@q
instance_normalization/ShapeShapeconv2d/BiasAdd:output:0*
T0*
_output_shapes
::��t
*instance_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,instance_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,instance_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$instance_normalization/strided_sliceStridedSlice%instance_normalization/Shape:output:03instance_normalization/strided_slice/stack:output:05instance_normalization/strided_slice/stack_1:output:05instance_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
,instance_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.instance_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.instance_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&instance_normalization/strided_slice_1StridedSlice%instance_normalization/Shape:output:05instance_normalization/strided_slice_1/stack:output:07instance_normalization/strided_slice_1/stack_1:output:07instance_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
,instance_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.instance_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.instance_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&instance_normalization/strided_slice_2StridedSlice%instance_normalization/Shape:output:05instance_normalization/strided_slice_2/stack:output:07instance_normalization/strided_slice_2/stack_1:output:07instance_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
,instance_normalization/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.instance_normalization/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.instance_normalization/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&instance_normalization/strided_slice_3StridedSlice%instance_normalization/Shape:output:05instance_normalization/strided_slice_3/stack:output:07instance_normalization/strided_slice_3/stack_1:output:07instance_normalization/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
5instance_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
#instance_normalization/moments/meanMeanconv2d/BiasAdd:output:0>instance_normalization/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������@*
	keep_dims(�
+instance_normalization/moments/StopGradientStopGradient,instance_normalization/moments/mean:output:0*
T0*/
_output_shapes
:���������@�
0instance_normalization/moments/SquaredDifferenceSquaredDifferenceconv2d/BiasAdd:output:04instance_normalization/moments/StopGradient:output:0*
T0*1
_output_shapes
:�����������@�
9instance_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
'instance_normalization/moments/varianceMean4instance_normalization/moments/SquaredDifference:z:0Binstance_normalization/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:���������@*
	keep_dims(�
-instance_normalization/Reshape/ReadVariableOpReadVariableOp6instance_normalization_reshape_readvariableop_resource*
_output_shapes
:@*
dtype0}
$instance_normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
instance_normalization/ReshapeReshape5instance_normalization/Reshape/ReadVariableOp:value:0-instance_normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:@�
/instance_normalization/Reshape_1/ReadVariableOpReadVariableOp8instance_normalization_reshape_1_readvariableop_resource*
_output_shapes
:@*
dtype0
&instance_normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
 instance_normalization/Reshape_1Reshape7instance_normalization/Reshape_1/ReadVariableOp:value:0/instance_normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:@k
&instance_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$instance_normalization/batchnorm/addAddV20instance_normalization/moments/variance:output:0/instance_normalization/batchnorm/add/y:output:0*
T0*/
_output_shapes
:���������@�
&instance_normalization/batchnorm/RsqrtRsqrt(instance_normalization/batchnorm/add:z:0*
T0*/
_output_shapes
:���������@�
$instance_normalization/batchnorm/mulMul*instance_normalization/batchnorm/Rsqrt:y:0'instance_normalization/Reshape:output:0*
T0*/
_output_shapes
:���������@�
&instance_normalization/batchnorm/mul_1Mulconv2d/BiasAdd:output:0(instance_normalization/batchnorm/mul:z:0*
T0*1
_output_shapes
:�����������@�
&instance_normalization/batchnorm/mul_2Mul,instance_normalization/moments/mean:output:0(instance_normalization/batchnorm/mul:z:0*
T0*/
_output_shapes
:���������@�
$instance_normalization/batchnorm/subSub)instance_normalization/Reshape_1:output:0*instance_normalization/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:���������@�
&instance_normalization/batchnorm/add_1AddV2*instance_normalization/batchnorm/mul_1:z:0(instance_normalization/batchnorm/sub:z:0*
T0*1
_output_shapes
:�����������@z

re_lu/ReluRelu*instance_normalization/batchnorm/add_1:z:0*
T0*1
_output_shapes
:�����������@�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_1/Conv2DConv2Dre_lu/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������*
paddingSAME*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������u
instance_normalization_1/ShapeShapeconv2d_1/BiasAdd:output:0*
T0*
_output_shapes
::��v
,instance_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.instance_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.instance_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&instance_normalization_1/strided_sliceStridedSlice'instance_normalization_1/Shape:output:05instance_normalization_1/strided_slice/stack:output:07instance_normalization_1/strided_slice/stack_1:output:07instance_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_1/strided_slice_1StridedSlice'instance_normalization_1/Shape:output:07instance_normalization_1/strided_slice_1/stack:output:09instance_normalization_1/strided_slice_1/stack_1:output:09instance_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_1/strided_slice_2StridedSlice'instance_normalization_1/Shape:output:07instance_normalization_1/strided_slice_2/stack:output:09instance_normalization_1/strided_slice_2/stack_1:output:09instance_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_1/strided_slice_3StridedSlice'instance_normalization_1/Shape:output:07instance_normalization_1/strided_slice_3/stack:output:09instance_normalization_1/strided_slice_3/stack_1:output:09instance_normalization_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
7instance_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
%instance_normalization_1/moments/meanMeanconv2d_1/BiasAdd:output:0@instance_normalization_1/moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
-instance_normalization_1/moments/StopGradientStopGradient.instance_normalization_1/moments/mean:output:0*
T0*0
_output_shapes
:�����������
2instance_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv2d_1/BiasAdd:output:06instance_normalization_1/moments/StopGradient:output:0*
T0*2
_output_shapes 
:�������������
;instance_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
)instance_normalization_1/moments/varianceMean6instance_normalization_1/moments/SquaredDifference:z:0Dinstance_normalization_1/moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
/instance_normalization_1/Reshape/ReadVariableOpReadVariableOp8instance_normalization_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0
&instance_normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
 instance_normalization_1/ReshapeReshape7instance_normalization_1/Reshape/ReadVariableOp:value:0/instance_normalization_1/Reshape/shape:output:0*
T0*'
_output_shapes
:��
1instance_normalization_1/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_1_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(instance_normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
"instance_normalization_1/Reshape_1Reshape9instance_normalization_1/Reshape_1/ReadVariableOp:value:01instance_normalization_1/Reshape_1/shape:output:0*
T0*'
_output_shapes
:�m
(instance_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&instance_normalization_1/batchnorm/addAddV22instance_normalization_1/moments/variance:output:01instance_normalization_1/batchnorm/add/y:output:0*
T0*0
_output_shapes
:�����������
(instance_normalization_1/batchnorm/RsqrtRsqrt*instance_normalization_1/batchnorm/add:z:0*
T0*0
_output_shapes
:�����������
&instance_normalization_1/batchnorm/mulMul,instance_normalization_1/batchnorm/Rsqrt:y:0)instance_normalization_1/Reshape:output:0*
T0*0
_output_shapes
:�����������
(instance_normalization_1/batchnorm/mul_1Mulconv2d_1/BiasAdd:output:0*instance_normalization_1/batchnorm/mul:z:0*
T0*2
_output_shapes 
:�������������
(instance_normalization_1/batchnorm/mul_2Mul.instance_normalization_1/moments/mean:output:0*instance_normalization_1/batchnorm/mul:z:0*
T0*0
_output_shapes
:�����������
&instance_normalization_1/batchnorm/subSub+instance_normalization_1/Reshape_1:output:0,instance_normalization_1/batchnorm/mul_2:z:0*
T0*0
_output_shapes
:�����������
(instance_normalization_1/batchnorm/add_1AddV2,instance_normalization_1/batchnorm/mul_1:z:0*instance_normalization_1/batchnorm/sub:z:0*
T0*2
_output_shapes 
:������������
re_lu_1/ReluRelu,instance_normalization_1/batchnorm/add_1:z:0*
T0*2
_output_shapes 
:�������������
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_2/Conv2DConv2Dre_lu_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�u
instance_normalization_2/ShapeShapeconv2d_2/BiasAdd:output:0*
T0*
_output_shapes
::��v
,instance_normalization_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.instance_normalization_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.instance_normalization_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&instance_normalization_2/strided_sliceStridedSlice'instance_normalization_2/Shape:output:05instance_normalization_2/strided_slice/stack:output:07instance_normalization_2/strided_slice/stack_1:output:07instance_normalization_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_2/strided_slice_1StridedSlice'instance_normalization_2/Shape:output:07instance_normalization_2/strided_slice_1/stack:output:09instance_normalization_2/strided_slice_1/stack_1:output:09instance_normalization_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_2/strided_slice_2StridedSlice'instance_normalization_2/Shape:output:07instance_normalization_2/strided_slice_2/stack:output:09instance_normalization_2/strided_slice_2/stack_1:output:09instance_normalization_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_2/strided_slice_3StridedSlice'instance_normalization_2/Shape:output:07instance_normalization_2/strided_slice_3/stack:output:09instance_normalization_2/strided_slice_3/stack_1:output:09instance_normalization_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
7instance_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
%instance_normalization_2/moments/meanMeanconv2d_2/BiasAdd:output:0@instance_normalization_2/moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
-instance_normalization_2/moments/StopGradientStopGradient.instance_normalization_2/moments/mean:output:0*
T0*0
_output_shapes
:�����������
2instance_normalization_2/moments/SquaredDifferenceSquaredDifferenceconv2d_2/BiasAdd:output:06instance_normalization_2/moments/StopGradient:output:0*
T0*0
_output_shapes
:���������@@��
;instance_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
)instance_normalization_2/moments/varianceMean6instance_normalization_2/moments/SquaredDifference:z:0Dinstance_normalization_2/moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
/instance_normalization_2/Reshape/ReadVariableOpReadVariableOp8instance_normalization_2_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0
&instance_normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
 instance_normalization_2/ReshapeReshape7instance_normalization_2/Reshape/ReadVariableOp:value:0/instance_normalization_2/Reshape/shape:output:0*
T0*'
_output_shapes
:��
1instance_normalization_2/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_2_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(instance_normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
"instance_normalization_2/Reshape_1Reshape9instance_normalization_2/Reshape_1/ReadVariableOp:value:01instance_normalization_2/Reshape_1/shape:output:0*
T0*'
_output_shapes
:�m
(instance_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&instance_normalization_2/batchnorm/addAddV22instance_normalization_2/moments/variance:output:01instance_normalization_2/batchnorm/add/y:output:0*
T0*0
_output_shapes
:�����������
(instance_normalization_2/batchnorm/RsqrtRsqrt*instance_normalization_2/batchnorm/add:z:0*
T0*0
_output_shapes
:�����������
&instance_normalization_2/batchnorm/mulMul,instance_normalization_2/batchnorm/Rsqrt:y:0)instance_normalization_2/Reshape:output:0*
T0*0
_output_shapes
:�����������
(instance_normalization_2/batchnorm/mul_1Mulconv2d_2/BiasAdd:output:0*instance_normalization_2/batchnorm/mul:z:0*
T0*0
_output_shapes
:���������@@��
(instance_normalization_2/batchnorm/mul_2Mul.instance_normalization_2/moments/mean:output:0*instance_normalization_2/batchnorm/mul:z:0*
T0*0
_output_shapes
:�����������
&instance_normalization_2/batchnorm/subSub+instance_normalization_2/Reshape_1:output:0,instance_normalization_2/batchnorm/mul_2:z:0*
T0*0
_output_shapes
:�����������
(instance_normalization_2/batchnorm/add_1AddV2,instance_normalization_2/batchnorm/mul_1:z:0*instance_normalization_2/batchnorm/sub:z:0*
T0*0
_output_shapes
:���������@@�}
re_lu_2/ReluRelu,instance_normalization_2/batchnorm/add_1:z:0*
T0*0
_output_shapes
:���������@@��
'residual/conv2d_3/Conv2D/ReadVariableOpReadVariableOp0residual_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual/conv2d_3/Conv2DConv2Dre_lu_2/Relu:activations:0/residual/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
(residual/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp1residual_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual/conv2d_3/BiasAddBiasAdd!residual/conv2d_3/Conv2D:output:00residual/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�}
residual/conv2d_3/ReluRelu"residual/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
'residual/conv2d_4/Conv2D/ReadVariableOpReadVariableOp0residual_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual/conv2d_4/Conv2DConv2D$residual/conv2d_3/Relu:activations:0/residual/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
(residual/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp1residual_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual/conv2d_4/BiasAddBiasAdd!residual/conv2d_4/Conv2D:output:00residual/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual/add/addAddV2"residual/conv2d_4/BiasAdd:output:0re_lu_2/Relu:activations:0*
T0*0
_output_shapes
:���������@@�l
residual/re_lu/ReluReluresidual/add/add:z:0*
T0*0
_output_shapes
:���������@@��
)residual_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp2residual_1_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_1/conv2d_5/Conv2DConv2D!residual/re_lu/Relu:activations:01residual_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
*residual_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp3residual_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_1/conv2d_5/BiasAddBiasAdd#residual_1/conv2d_5/Conv2D:output:02residual_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_1/conv2d_5/ReluRelu$residual_1/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
)residual_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp2residual_1_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_1/conv2d_6/Conv2DConv2D&residual_1/conv2d_5/Relu:activations:01residual_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
*residual_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp3residual_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_1/conv2d_6/BiasAddBiasAdd#residual_1/conv2d_6/Conv2D:output:02residual_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_1/add_1/addAddV2$residual_1/conv2d_6/BiasAdd:output:0!residual/re_lu/Relu:activations:0*
T0*0
_output_shapes
:���������@@�t
residual_1/re_lu_1/ReluReluresidual_1/add_1/add:z:0*
T0*0
_output_shapes
:���������@@��
)residual_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp2residual_2_conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_2/conv2d_7/Conv2DConv2D%residual_1/re_lu_1/Relu:activations:01residual_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
*residual_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp3residual_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_2/conv2d_7/BiasAddBiasAdd#residual_2/conv2d_7/Conv2D:output:02residual_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_2/conv2d_7/ReluRelu$residual_2/conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
)residual_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp2residual_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_2/conv2d_8/Conv2DConv2D&residual_2/conv2d_7/Relu:activations:01residual_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
*residual_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp3residual_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_2/conv2d_8/BiasAddBiasAdd#residual_2/conv2d_8/Conv2D:output:02residual_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_2/add_2/addAddV2$residual_2/conv2d_8/BiasAdd:output:0%residual_1/re_lu_1/Relu:activations:0*
T0*0
_output_shapes
:���������@@�t
residual_2/re_lu_2/ReluReluresidual_2/add_2/add:z:0*
T0*0
_output_shapes
:���������@@��
)residual_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp2residual_3_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_3/conv2d_9/Conv2DConv2D%residual_2/re_lu_2/Relu:activations:01residual_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
*residual_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp3residual_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_3/conv2d_9/BiasAddBiasAdd#residual_3/conv2d_9/Conv2D:output:02residual_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_3/conv2d_9/ReluRelu$residual_3/conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
*residual_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp3residual_3_conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_3/conv2d_10/Conv2DConv2D&residual_3/conv2d_9/Relu:activations:02residual_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp4residual_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_3/conv2d_10/BiasAddBiasAdd$residual_3/conv2d_10/Conv2D:output:03residual_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_3/add_3/addAddV2%residual_3/conv2d_10/BiasAdd:output:0%residual_2/re_lu_2/Relu:activations:0*
T0*0
_output_shapes
:���������@@�t
residual_3/re_lu_3/ReluReluresidual_3/add_3/add:z:0*
T0*0
_output_shapes
:���������@@��
*residual_4/conv2d_11/Conv2D/ReadVariableOpReadVariableOp3residual_4_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_4/conv2d_11/Conv2DConv2D%residual_3/re_lu_3/Relu:activations:02residual_4/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_4/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp4residual_4_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_4/conv2d_11/BiasAddBiasAdd$residual_4/conv2d_11/Conv2D:output:03residual_4/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_4/conv2d_11/ReluRelu%residual_4/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
*residual_4/conv2d_12/Conv2D/ReadVariableOpReadVariableOp3residual_4_conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_4/conv2d_12/Conv2DConv2D'residual_4/conv2d_11/Relu:activations:02residual_4/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_4/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp4residual_4_conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_4/conv2d_12/BiasAddBiasAdd$residual_4/conv2d_12/Conv2D:output:03residual_4/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_4/add_4/addAddV2%residual_4/conv2d_12/BiasAdd:output:0%residual_3/re_lu_3/Relu:activations:0*
T0*0
_output_shapes
:���������@@�t
residual_4/re_lu_4/ReluReluresidual_4/add_4/add:z:0*
T0*0
_output_shapes
:���������@@��
*residual_5/conv2d_13/Conv2D/ReadVariableOpReadVariableOp3residual_5_conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_5/conv2d_13/Conv2DConv2D%residual_4/re_lu_4/Relu:activations:02residual_5/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_5/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp4residual_5_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_5/conv2d_13/BiasAddBiasAdd$residual_5/conv2d_13/Conv2D:output:03residual_5/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_5/conv2d_13/ReluRelu%residual_5/conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
*residual_5/conv2d_14/Conv2D/ReadVariableOpReadVariableOp3residual_5_conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_5/conv2d_14/Conv2DConv2D'residual_5/conv2d_13/Relu:activations:02residual_5/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_5/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp4residual_5_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_5/conv2d_14/BiasAddBiasAdd$residual_5/conv2d_14/Conv2D:output:03residual_5/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_5/add_5/addAddV2%residual_5/conv2d_14/BiasAdd:output:0%residual_4/re_lu_4/Relu:activations:0*
T0*0
_output_shapes
:���������@@�t
residual_5/re_lu_5/ReluReluresidual_5/add_5/add:z:0*
T0*0
_output_shapes
:���������@@��
*residual_6/conv2d_15/Conv2D/ReadVariableOpReadVariableOp3residual_6_conv2d_15_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_6/conv2d_15/Conv2DConv2D%residual_5/re_lu_5/Relu:activations:02residual_6/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_6/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp4residual_6_conv2d_15_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_6/conv2d_15/BiasAddBiasAdd$residual_6/conv2d_15/Conv2D:output:03residual_6/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_6/conv2d_15/ReluRelu%residual_6/conv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
*residual_6/conv2d_16/Conv2D/ReadVariableOpReadVariableOp3residual_6_conv2d_16_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_6/conv2d_16/Conv2DConv2D'residual_6/conv2d_15/Relu:activations:02residual_6/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_6/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp4residual_6_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_6/conv2d_16/BiasAddBiasAdd$residual_6/conv2d_16/Conv2D:output:03residual_6/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_6/add_6/addAddV2%residual_6/conv2d_16/BiasAdd:output:0%residual_5/re_lu_5/Relu:activations:0*
T0*0
_output_shapes
:���������@@�t
residual_6/re_lu_6/ReluReluresidual_6/add_6/add:z:0*
T0*0
_output_shapes
:���������@@��
*residual_7/conv2d_17/Conv2D/ReadVariableOpReadVariableOp3residual_7_conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_7/conv2d_17/Conv2DConv2D%residual_6/re_lu_6/Relu:activations:02residual_7/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_7/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp4residual_7_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_7/conv2d_17/BiasAddBiasAdd$residual_7/conv2d_17/Conv2D:output:03residual_7/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_7/conv2d_17/ReluRelu%residual_7/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
*residual_7/conv2d_18/Conv2D/ReadVariableOpReadVariableOp3residual_7_conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_7/conv2d_18/Conv2DConv2D'residual_7/conv2d_17/Relu:activations:02residual_7/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_7/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp4residual_7_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_7/conv2d_18/BiasAddBiasAdd$residual_7/conv2d_18/Conv2D:output:03residual_7/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_7/add_7/addAddV2%residual_7/conv2d_18/BiasAdd:output:0%residual_6/re_lu_6/Relu:activations:0*
T0*0
_output_shapes
:���������@@�t
residual_7/re_lu_7/ReluReluresidual_7/add_7/add:z:0*
T0*0
_output_shapes
:���������@@��
*residual_8/conv2d_19/Conv2D/ReadVariableOpReadVariableOp3residual_8_conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_8/conv2d_19/Conv2DConv2D%residual_7/re_lu_7/Relu:activations:02residual_8/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_8/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp4residual_8_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_8/conv2d_19/BiasAddBiasAdd$residual_8/conv2d_19/Conv2D:output:03residual_8/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_8/conv2d_19/ReluRelu%residual_8/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
*residual_8/conv2d_20/Conv2D/ReadVariableOpReadVariableOp3residual_8_conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_8/conv2d_20/Conv2DConv2D'residual_8/conv2d_19/Relu:activations:02residual_8/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_8/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp4residual_8_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_8/conv2d_20/BiasAddBiasAdd$residual_8/conv2d_20/Conv2D:output:03residual_8/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_8/add_8/addAddV2%residual_8/conv2d_20/BiasAdd:output:0%residual_7/re_lu_7/Relu:activations:0*
T0*0
_output_shapes
:���������@@�t
residual_8/re_lu_8/ReluReluresidual_8/add_8/add:z:0*
T0*0
_output_shapes
:���������@@�y
conv2d_transpose/ShapeShape%residual_8/re_lu_8/Relu:activations:0*
T0*
_output_shapes
::��n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�[
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�[
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%residual_8/re_lu_8/Relu:activations:0*
T0*2
_output_shapes 
:������������*
paddingSAME*
strides
�
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������}
instance_normalization_3/ShapeShape!conv2d_transpose/BiasAdd:output:0*
T0*
_output_shapes
::��v
,instance_normalization_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.instance_normalization_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.instance_normalization_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&instance_normalization_3/strided_sliceStridedSlice'instance_normalization_3/Shape:output:05instance_normalization_3/strided_slice/stack:output:07instance_normalization_3/strided_slice/stack_1:output:07instance_normalization_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_3/strided_slice_1StridedSlice'instance_normalization_3/Shape:output:07instance_normalization_3/strided_slice_1/stack:output:09instance_normalization_3/strided_slice_1/stack_1:output:09instance_normalization_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_3/strided_slice_2StridedSlice'instance_normalization_3/Shape:output:07instance_normalization_3/strided_slice_2/stack:output:09instance_normalization_3/strided_slice_2/stack_1:output:09instance_normalization_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_3/strided_slice_3StridedSlice'instance_normalization_3/Shape:output:07instance_normalization_3/strided_slice_3/stack:output:09instance_normalization_3/strided_slice_3/stack_1:output:09instance_normalization_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
7instance_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
%instance_normalization_3/moments/meanMean!conv2d_transpose/BiasAdd:output:0@instance_normalization_3/moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
-instance_normalization_3/moments/StopGradientStopGradient.instance_normalization_3/moments/mean:output:0*
T0*0
_output_shapes
:�����������
2instance_normalization_3/moments/SquaredDifferenceSquaredDifference!conv2d_transpose/BiasAdd:output:06instance_normalization_3/moments/StopGradient:output:0*
T0*2
_output_shapes 
:�������������
;instance_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
)instance_normalization_3/moments/varianceMean6instance_normalization_3/moments/SquaredDifference:z:0Dinstance_normalization_3/moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
/instance_normalization_3/Reshape/ReadVariableOpReadVariableOp8instance_normalization_3_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0
&instance_normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
 instance_normalization_3/ReshapeReshape7instance_normalization_3/Reshape/ReadVariableOp:value:0/instance_normalization_3/Reshape/shape:output:0*
T0*'
_output_shapes
:��
1instance_normalization_3/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_3_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(instance_normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
"instance_normalization_3/Reshape_1Reshape9instance_normalization_3/Reshape_1/ReadVariableOp:value:01instance_normalization_3/Reshape_1/shape:output:0*
T0*'
_output_shapes
:�m
(instance_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&instance_normalization_3/batchnorm/addAddV22instance_normalization_3/moments/variance:output:01instance_normalization_3/batchnorm/add/y:output:0*
T0*0
_output_shapes
:�����������
(instance_normalization_3/batchnorm/RsqrtRsqrt*instance_normalization_3/batchnorm/add:z:0*
T0*0
_output_shapes
:�����������
&instance_normalization_3/batchnorm/mulMul,instance_normalization_3/batchnorm/Rsqrt:y:0)instance_normalization_3/Reshape:output:0*
T0*0
_output_shapes
:�����������
(instance_normalization_3/batchnorm/mul_1Mul!conv2d_transpose/BiasAdd:output:0*instance_normalization_3/batchnorm/mul:z:0*
T0*2
_output_shapes 
:�������������
(instance_normalization_3/batchnorm/mul_2Mul.instance_normalization_3/moments/mean:output:0*instance_normalization_3/batchnorm/mul:z:0*
T0*0
_output_shapes
:�����������
&instance_normalization_3/batchnorm/subSub+instance_normalization_3/Reshape_1:output:0,instance_normalization_3/batchnorm/mul_2:z:0*
T0*0
_output_shapes
:�����������
(instance_normalization_3/batchnorm/add_1AddV2,instance_normalization_3/batchnorm/mul_1:z:0*instance_normalization_3/batchnorm/sub:z:0*
T0*2
_output_shapes 
:������������
re_lu_3/ReluRelu,instance_normalization_3/batchnorm/add_1:z:0*
T0*2
_output_shapes 
:������������p
conv2d_transpose_1/ShapeShapere_lu_3/Relu:activations:0*
T0*
_output_shapes
::��p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�]
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0re_lu_3/Relu:activations:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@
instance_normalization_4/ShapeShape#conv2d_transpose_1/BiasAdd:output:0*
T0*
_output_shapes
::��v
,instance_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.instance_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.instance_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&instance_normalization_4/strided_sliceStridedSlice'instance_normalization_4/Shape:output:05instance_normalization_4/strided_slice/stack:output:07instance_normalization_4/strided_slice/stack_1:output:07instance_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_4/strided_slice_1StridedSlice'instance_normalization_4/Shape:output:07instance_normalization_4/strided_slice_1/stack:output:09instance_normalization_4/strided_slice_1/stack_1:output:09instance_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_4/strided_slice_2StridedSlice'instance_normalization_4/Shape:output:07instance_normalization_4/strided_slice_2/stack:output:09instance_normalization_4/strided_slice_2/stack_1:output:09instance_normalization_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_4/strided_slice_3StridedSlice'instance_normalization_4/Shape:output:07instance_normalization_4/strided_slice_3/stack:output:09instance_normalization_4/strided_slice_3/stack_1:output:09instance_normalization_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
7instance_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
%instance_normalization_4/moments/meanMean#conv2d_transpose_1/BiasAdd:output:0@instance_normalization_4/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������@*
	keep_dims(�
-instance_normalization_4/moments/StopGradientStopGradient.instance_normalization_4/moments/mean:output:0*
T0*/
_output_shapes
:���������@�
2instance_normalization_4/moments/SquaredDifferenceSquaredDifference#conv2d_transpose_1/BiasAdd:output:06instance_normalization_4/moments/StopGradient:output:0*
T0*1
_output_shapes
:�����������@�
;instance_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
)instance_normalization_4/moments/varianceMean6instance_normalization_4/moments/SquaredDifference:z:0Dinstance_normalization_4/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:���������@*
	keep_dims(�
/instance_normalization_4/Reshape/ReadVariableOpReadVariableOp8instance_normalization_4_reshape_readvariableop_resource*
_output_shapes
:@*
dtype0
&instance_normalization_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
 instance_normalization_4/ReshapeReshape7instance_normalization_4/Reshape/ReadVariableOp:value:0/instance_normalization_4/Reshape/shape:output:0*
T0*&
_output_shapes
:@�
1instance_normalization_4/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_4_reshape_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
(instance_normalization_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
"instance_normalization_4/Reshape_1Reshape9instance_normalization_4/Reshape_1/ReadVariableOp:value:01instance_normalization_4/Reshape_1/shape:output:0*
T0*&
_output_shapes
:@m
(instance_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&instance_normalization_4/batchnorm/addAddV22instance_normalization_4/moments/variance:output:01instance_normalization_4/batchnorm/add/y:output:0*
T0*/
_output_shapes
:���������@�
(instance_normalization_4/batchnorm/RsqrtRsqrt*instance_normalization_4/batchnorm/add:z:0*
T0*/
_output_shapes
:���������@�
&instance_normalization_4/batchnorm/mulMul,instance_normalization_4/batchnorm/Rsqrt:y:0)instance_normalization_4/Reshape:output:0*
T0*/
_output_shapes
:���������@�
(instance_normalization_4/batchnorm/mul_1Mul#conv2d_transpose_1/BiasAdd:output:0*instance_normalization_4/batchnorm/mul:z:0*
T0*1
_output_shapes
:�����������@�
(instance_normalization_4/batchnorm/mul_2Mul.instance_normalization_4/moments/mean:output:0*instance_normalization_4/batchnorm/mul:z:0*
T0*/
_output_shapes
:���������@�
&instance_normalization_4/batchnorm/subSub+instance_normalization_4/Reshape_1:output:0,instance_normalization_4/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:���������@�
(instance_normalization_4/batchnorm/add_1AddV2,instance_normalization_4/batchnorm/mul_1:z:0*instance_normalization_4/batchnorm/sub:z:0*
T0*1
_output_shapes
:�����������@~
re_lu_4/ReluRelu,instance_normalization_4/batchnorm/add_1:z:0*
T0*1
_output_shapes
:�����������@�
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d_21/Conv2DConv2Dre_lu_4/Relu:activations:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������n
conv2d_21/TanhTanhconv2d_21/BiasAdd:output:0*
T0*1
_output_shapes
:�����������k
IdentityIdentityconv2d_21/Tanh:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp.^instance_normalization/Reshape/ReadVariableOp0^instance_normalization/Reshape_1/ReadVariableOp0^instance_normalization_1/Reshape/ReadVariableOp2^instance_normalization_1/Reshape_1/ReadVariableOp0^instance_normalization_2/Reshape/ReadVariableOp2^instance_normalization_2/Reshape_1/ReadVariableOp0^instance_normalization_3/Reshape/ReadVariableOp2^instance_normalization_3/Reshape_1/ReadVariableOp0^instance_normalization_4/Reshape/ReadVariableOp2^instance_normalization_4/Reshape_1/ReadVariableOp)^residual/conv2d_3/BiasAdd/ReadVariableOp(^residual/conv2d_3/Conv2D/ReadVariableOp)^residual/conv2d_4/BiasAdd/ReadVariableOp(^residual/conv2d_4/Conv2D/ReadVariableOp+^residual_1/conv2d_5/BiasAdd/ReadVariableOp*^residual_1/conv2d_5/Conv2D/ReadVariableOp+^residual_1/conv2d_6/BiasAdd/ReadVariableOp*^residual_1/conv2d_6/Conv2D/ReadVariableOp+^residual_2/conv2d_7/BiasAdd/ReadVariableOp*^residual_2/conv2d_7/Conv2D/ReadVariableOp+^residual_2/conv2d_8/BiasAdd/ReadVariableOp*^residual_2/conv2d_8/Conv2D/ReadVariableOp,^residual_3/conv2d_10/BiasAdd/ReadVariableOp+^residual_3/conv2d_10/Conv2D/ReadVariableOp+^residual_3/conv2d_9/BiasAdd/ReadVariableOp*^residual_3/conv2d_9/Conv2D/ReadVariableOp,^residual_4/conv2d_11/BiasAdd/ReadVariableOp+^residual_4/conv2d_11/Conv2D/ReadVariableOp,^residual_4/conv2d_12/BiasAdd/ReadVariableOp+^residual_4/conv2d_12/Conv2D/ReadVariableOp,^residual_5/conv2d_13/BiasAdd/ReadVariableOp+^residual_5/conv2d_13/Conv2D/ReadVariableOp,^residual_5/conv2d_14/BiasAdd/ReadVariableOp+^residual_5/conv2d_14/Conv2D/ReadVariableOp,^residual_6/conv2d_15/BiasAdd/ReadVariableOp+^residual_6/conv2d_15/Conv2D/ReadVariableOp,^residual_6/conv2d_16/BiasAdd/ReadVariableOp+^residual_6/conv2d_16/Conv2D/ReadVariableOp,^residual_7/conv2d_17/BiasAdd/ReadVariableOp+^residual_7/conv2d_17/Conv2D/ReadVariableOp,^residual_7/conv2d_18/BiasAdd/ReadVariableOp+^residual_7/conv2d_18/Conv2D/ReadVariableOp,^residual_8/conv2d_19/BiasAdd/ReadVariableOp+^residual_8/conv2d_19/Conv2D/ReadVariableOp,^residual_8/conv2d_20/BiasAdd/ReadVariableOp+^residual_8/conv2d_20/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2^
-instance_normalization/Reshape/ReadVariableOp-instance_normalization/Reshape/ReadVariableOp2b
/instance_normalization/Reshape_1/ReadVariableOp/instance_normalization/Reshape_1/ReadVariableOp2b
/instance_normalization_1/Reshape/ReadVariableOp/instance_normalization_1/Reshape/ReadVariableOp2f
1instance_normalization_1/Reshape_1/ReadVariableOp1instance_normalization_1/Reshape_1/ReadVariableOp2b
/instance_normalization_2/Reshape/ReadVariableOp/instance_normalization_2/Reshape/ReadVariableOp2f
1instance_normalization_2/Reshape_1/ReadVariableOp1instance_normalization_2/Reshape_1/ReadVariableOp2b
/instance_normalization_3/Reshape/ReadVariableOp/instance_normalization_3/Reshape/ReadVariableOp2f
1instance_normalization_3/Reshape_1/ReadVariableOp1instance_normalization_3/Reshape_1/ReadVariableOp2b
/instance_normalization_4/Reshape/ReadVariableOp/instance_normalization_4/Reshape/ReadVariableOp2f
1instance_normalization_4/Reshape_1/ReadVariableOp1instance_normalization_4/Reshape_1/ReadVariableOp2T
(residual/conv2d_3/BiasAdd/ReadVariableOp(residual/conv2d_3/BiasAdd/ReadVariableOp2R
'residual/conv2d_3/Conv2D/ReadVariableOp'residual/conv2d_3/Conv2D/ReadVariableOp2T
(residual/conv2d_4/BiasAdd/ReadVariableOp(residual/conv2d_4/BiasAdd/ReadVariableOp2R
'residual/conv2d_4/Conv2D/ReadVariableOp'residual/conv2d_4/Conv2D/ReadVariableOp2X
*residual_1/conv2d_5/BiasAdd/ReadVariableOp*residual_1/conv2d_5/BiasAdd/ReadVariableOp2V
)residual_1/conv2d_5/Conv2D/ReadVariableOp)residual_1/conv2d_5/Conv2D/ReadVariableOp2X
*residual_1/conv2d_6/BiasAdd/ReadVariableOp*residual_1/conv2d_6/BiasAdd/ReadVariableOp2V
)residual_1/conv2d_6/Conv2D/ReadVariableOp)residual_1/conv2d_6/Conv2D/ReadVariableOp2X
*residual_2/conv2d_7/BiasAdd/ReadVariableOp*residual_2/conv2d_7/BiasAdd/ReadVariableOp2V
)residual_2/conv2d_7/Conv2D/ReadVariableOp)residual_2/conv2d_7/Conv2D/ReadVariableOp2X
*residual_2/conv2d_8/BiasAdd/ReadVariableOp*residual_2/conv2d_8/BiasAdd/ReadVariableOp2V
)residual_2/conv2d_8/Conv2D/ReadVariableOp)residual_2/conv2d_8/Conv2D/ReadVariableOp2Z
+residual_3/conv2d_10/BiasAdd/ReadVariableOp+residual_3/conv2d_10/BiasAdd/ReadVariableOp2X
*residual_3/conv2d_10/Conv2D/ReadVariableOp*residual_3/conv2d_10/Conv2D/ReadVariableOp2X
*residual_3/conv2d_9/BiasAdd/ReadVariableOp*residual_3/conv2d_9/BiasAdd/ReadVariableOp2V
)residual_3/conv2d_9/Conv2D/ReadVariableOp)residual_3/conv2d_9/Conv2D/ReadVariableOp2Z
+residual_4/conv2d_11/BiasAdd/ReadVariableOp+residual_4/conv2d_11/BiasAdd/ReadVariableOp2X
*residual_4/conv2d_11/Conv2D/ReadVariableOp*residual_4/conv2d_11/Conv2D/ReadVariableOp2Z
+residual_4/conv2d_12/BiasAdd/ReadVariableOp+residual_4/conv2d_12/BiasAdd/ReadVariableOp2X
*residual_4/conv2d_12/Conv2D/ReadVariableOp*residual_4/conv2d_12/Conv2D/ReadVariableOp2Z
+residual_5/conv2d_13/BiasAdd/ReadVariableOp+residual_5/conv2d_13/BiasAdd/ReadVariableOp2X
*residual_5/conv2d_13/Conv2D/ReadVariableOp*residual_5/conv2d_13/Conv2D/ReadVariableOp2Z
+residual_5/conv2d_14/BiasAdd/ReadVariableOp+residual_5/conv2d_14/BiasAdd/ReadVariableOp2X
*residual_5/conv2d_14/Conv2D/ReadVariableOp*residual_5/conv2d_14/Conv2D/ReadVariableOp2Z
+residual_6/conv2d_15/BiasAdd/ReadVariableOp+residual_6/conv2d_15/BiasAdd/ReadVariableOp2X
*residual_6/conv2d_15/Conv2D/ReadVariableOp*residual_6/conv2d_15/Conv2D/ReadVariableOp2Z
+residual_6/conv2d_16/BiasAdd/ReadVariableOp+residual_6/conv2d_16/BiasAdd/ReadVariableOp2X
*residual_6/conv2d_16/Conv2D/ReadVariableOp*residual_6/conv2d_16/Conv2D/ReadVariableOp2Z
+residual_7/conv2d_17/BiasAdd/ReadVariableOp+residual_7/conv2d_17/BiasAdd/ReadVariableOp2X
*residual_7/conv2d_17/Conv2D/ReadVariableOp*residual_7/conv2d_17/Conv2D/ReadVariableOp2Z
+residual_7/conv2d_18/BiasAdd/ReadVariableOp+residual_7/conv2d_18/BiasAdd/ReadVariableOp2X
*residual_7/conv2d_18/Conv2D/ReadVariableOp*residual_7/conv2d_18/Conv2D/ReadVariableOp2Z
+residual_8/conv2d_19/BiasAdd/ReadVariableOp+residual_8/conv2d_19/BiasAdd/ReadVariableOp2X
*residual_8/conv2d_19/Conv2D/ReadVariableOp*residual_8/conv2d_19/Conv2D/ReadVariableOp2Z
+residual_8/conv2d_20/BiasAdd/ReadVariableOp+residual_8/conv2d_20/BiasAdd/ReadVariableOp2X
*residual_8/conv2d_20/Conv2D/ReadVariableOp*residual_8/conv2d_20/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
D__inference_conv2d_4_layer_call_and_return_conditional_losses_740708

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
+__inference_residual_5_layer_call_fn_744497

inputs#
unknown:��
	unknown_0:	�%
	unknown_1:��
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_5_layer_call_and_return_conditional_losses_741087x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
D
(__inference_re_lu_3_layer_call_fn_744711

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_741791k
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������:Z V
2
_output_shapes 
:������������
 
_user_specified_nameinputs
�*
�
R__inference_instance_normalization_layer_call_and_return_conditional_losses_744152

inputs-
reshape_readvariableop_resource:@/
!reshape_1_readvariableop_resource:@
identity��Reshape/ReadVariableOp�Reshape_1/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������@*
	keep_dims(u
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:���������@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*1
_output_shapes
:�����������@s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:���������@*
	keep_dims(r
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:@*
dtype0f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   {
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:@v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:@*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:@T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:���������@e
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:���������@u
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:���������@m
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*1
_output_shapes
:�����������@z
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:���������@w
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:���������@|
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*1
_output_shapes
:�����������@l
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*1
_output_shapes
:�����������@z
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�*
�
T__inference_instance_normalization_3_layer_call_and_return_conditional_losses_744706

inputs.
reshape_readvariableop_resource:	�0
!reshape_1_readvariableop_resource:	�
identity��Reshape/ReadVariableOp�Reshape_1/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(v
moments/StopGradientStopGradientmoments/mean:output:0*
T0*0
_output_shapes
:�����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*2
_output_shapes 
:������������s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(s
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes	
:�*
dtype0f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   |
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*'
_output_shapes
:�w
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:�T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*0
_output_shapes
:����������f
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*0
_output_shapes
:����������v
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*0
_output_shapes
:����������n
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*2
_output_shapes 
:������������{
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*0
_output_shapes
:����������x
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*0
_output_shapes
:����������}
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*2
_output_shapes 
:������������m
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*2
_output_shapes 
:������������z
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":������������: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:Z V
2
_output_shapes 
:������������
 
_user_specified_nameinputs
�

�
D__inference_conv2d_8_layer_call_and_return_conditional_losses_740856

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
)__inference_conv2d_7_layer_call_fn_744927

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_740840x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
D
(__inference_re_lu_4_layer_call_fn_744815

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_741852j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:�����������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������@:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�*
�
T__inference_instance_normalization_4_layer_call_and_return_conditional_losses_744810

inputs-
reshape_readvariableop_resource:@/
!reshape_1_readvariableop_resource:@
identity��Reshape/ReadVariableOp�Reshape_1/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������@*
	keep_dims(u
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:���������@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*1
_output_shapes
:�����������@s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:���������@*
	keep_dims(r
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:@*
dtype0f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   {
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:@v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:@*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:@T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:���������@e
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:���������@u
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:���������@m
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*1
_output_shapes
:�����������@z
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:���������@w
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:���������@|
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*1
_output_shapes
:�����������@l
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*1
_output_shapes
:�����������@z
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
)__inference_conv2d_2_layer_call_fn_744252

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_741589x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:������������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_15_layer_call_and_return_conditional_losses_745094

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������@@�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
D__inference_residual_layer_call_and_return_conditional_losses_740717

inputs+
conv2d_3_740693:��
conv2d_3_740695:	�+
conv2d_4_740709:��
conv2d_4_740711:	�
identity�� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_740693conv2d_3_740695*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_740692�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_740709conv2d_4_740711*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_740708~
add/addAddV2)conv2d_4/StatefulPartitionedCall:output:0inputs*
T0*0
_output_shapes
:���������@@�Z

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:���������@@�p
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@��
NoOpNoOp!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�*
�
T__inference_instance_normalization_1_layer_call_and_return_conditional_losses_744233

inputs.
reshape_readvariableop_resource:	�0
!reshape_1_readvariableop_resource:	�
identity��Reshape/ReadVariableOp�Reshape_1/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(v
moments/StopGradientStopGradientmoments/mean:output:0*
T0*0
_output_shapes
:�����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*2
_output_shapes 
:������������s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(s
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes	
:�*
dtype0f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   |
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*'
_output_shapes
:�w
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:�T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*0
_output_shapes
:����������f
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*0
_output_shapes
:����������v
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*0
_output_shapes
:����������n
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*2
_output_shapes 
:������������{
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*0
_output_shapes
:����������x
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*0
_output_shapes
:����������}
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*2
_output_shapes 
:������������m
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*2
_output_shapes 
:������������z
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":������������: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:Z V
2
_output_shapes 
:������������
 
_user_specified_nameinputs
�
_
C__inference_re_lu_1_layer_call_and_return_conditional_losses_741577

inputs
identityQ
ReluReluinputs*
T0*2
_output_shapes 
:������������e
IdentityIdentityRelu:activations:0*
T0*2
_output_shapes 
:������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������:Z V
2
_output_shapes 
:������������
 
_user_specified_nameinputs
�
�
7__inference_instance_normalization_layer_call_fn_744109

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *[
fVRT
R__inference_instance_normalization_layer_call_and_return_conditional_losses_741494y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
)__inference_conv2d_4_layer_call_fn_744869

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_740708x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
*__inference_conv2d_17_layer_call_fn_745122

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_741210x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
+__inference_sequential_layer_call_fn_742550
input_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�&

unknown_25:��

unknown_26:	�&

unknown_27:��

unknown_28:	�&

unknown_29:��

unknown_30:	�&

unknown_31:��

unknown_32:	�&

unknown_33:��

unknown_34:	�&

unknown_35:��

unknown_36:	�&

unknown_37:��

unknown_38:	�&

unknown_39:��

unknown_40:	�&

unknown_41:��

unknown_42:	�&

unknown_43:��

unknown_44:	�&

unknown_45:��

unknown_46:	�&

unknown_47:��

unknown_48:	�

unknown_49:	�

unknown_50:	�%

unknown_51:@�

unknown_52:@

unknown_53:@

unknown_54:@$

unknown_55:@

unknown_56:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_742431y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
)__inference_residual_layer_call_fn_744337

inputs#
unknown:��
	unknown_0:	�%
	unknown_1:��
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_residual_layer_call_and_return_conditional_losses_740717x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
+__inference_residual_4_layer_call_fn_744465

inputs#
unknown:��
	unknown_0:	�%
	unknown_1:��
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_4_layer_call_and_return_conditional_losses_741013x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�*
�
R__inference_instance_normalization_layer_call_and_return_conditional_losses_741494

inputs-
reshape_readvariableop_resource:@/
!reshape_1_readvariableop_resource:@
identity��Reshape/ReadVariableOp�Reshape_1/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������@*
	keep_dims(u
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:���������@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*1
_output_shapes
:�����������@s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:���������@*
	keep_dims(r
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:@*
dtype0f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   {
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:@v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:@*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:@T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:���������@e
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:���������@u
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:���������@m
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*1
_output_shapes
:�����������@z
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:���������@w
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:���������@|
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*1
_output_shapes
:�����������@l
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*1
_output_shapes
:�����������@z
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�*
�
T__inference_instance_normalization_2_layer_call_and_return_conditional_losses_741638

inputs.
reshape_readvariableop_resource:	�0
!reshape_1_readvariableop_resource:	�
identity��Reshape/ReadVariableOp�Reshape_1/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(v
moments/StopGradientStopGradientmoments/mean:output:0*
T0*0
_output_shapes
:�����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:���������@@�s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(s
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes	
:�*
dtype0f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            |
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*'
_output_shapes
:�w
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:�T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*0
_output_shapes
:����������f
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*0
_output_shapes
:����������v
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*0
_output_shapes
:����������l
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*0
_output_shapes
:���������@@�{
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*0
_output_shapes
:����������x
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*0
_output_shapes
:����������{
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*0
_output_shapes
:���������@@�k
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*0
_output_shapes
:���������@@�z
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�

�
E__inference_conv2d_12_layer_call_and_return_conditional_losses_741004

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�

�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_744181

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������j
IdentityIdentityBiasAdd:output:0^NoOp*
T0*2
_output_shapes 
:������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
D__inference_conv2d_9_layer_call_and_return_conditional_losses_740914

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������@@�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_744860

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������@@�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
E__inference_conv2d_21_layer_call_and_return_conditional_losses_744840

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������Z
TanhTanhBiasAdd:output:0*
T0*1
_output_shapes
:�����������a
IdentityIdentityTanh:y:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
9__inference_instance_normalization_2_layer_call_fn_744271

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *]
fXRV
T__inference_instance_normalization_2_layer_call_and_return_conditional_losses_741638x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�*
�
T__inference_instance_normalization_2_layer_call_and_return_conditional_losses_744314

inputs.
reshape_readvariableop_resource:	�0
!reshape_1_readvariableop_resource:	�
identity��Reshape/ReadVariableOp�Reshape_1/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(v
moments/StopGradientStopGradientmoments/mean:output:0*
T0*0
_output_shapes
:�����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:���������@@�s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(s
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes	
:�*
dtype0f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            |
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*'
_output_shapes
:�w
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:�T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*0
_output_shapes
:����������f
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*0
_output_shapes
:����������v
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*0
_output_shapes
:����������l
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*0
_output_shapes
:���������@@�{
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*0
_output_shapes
:����������x
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*0
_output_shapes
:����������{
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*0
_output_shapes
:���������@@�k
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*0
_output_shapes
:���������@@�z
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
E__inference_conv2d_17_layer_call_and_return_conditional_losses_745133

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������@@�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
F__inference_residual_8_layer_call_and_return_conditional_losses_741309

inputs,
conv2d_19_741285:��
conv2d_19_741287:	�,
conv2d_20_741301:��
conv2d_20_741303:	�
identity��!conv2d_19/StatefulPartitionedCall�!conv2d_20/StatefulPartitionedCall�
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_19_741285conv2d_19_741287*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_741284�
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0conv2d_20_741301conv2d_20_741303*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_741300
add/addAddV2*conv2d_20/StatefulPartitionedCall:output:0inputs*
T0*0
_output_shapes
:���������@@�Z

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:���������@@�p
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@��
NoOpNoOp"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
D__inference_conv2d_7_layer_call_and_return_conditional_losses_744938

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������@@�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
)__inference_conv2d_8_layer_call_fn_744947

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_740856x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�

�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_741589

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:������������
 
_user_specified_nameinputs
�

�
B__inference_conv2d_layer_call_and_return_conditional_losses_744100

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:�����������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
_
C__inference_re_lu_4_layer_call_and_return_conditional_losses_744820

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:�����������@d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:�����������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������@:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�

�
E__inference_conv2d_10_layer_call_and_return_conditional_losses_744996

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�

�
D__inference_conv2d_4_layer_call_and_return_conditional_losses_744879

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
� 
�
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_741377

inputsD
(conv2d_transpose_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_740692

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������@@�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�

�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_744262

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:������������
 
_user_specified_nameinputs
��
�
F__inference_sequential_layer_call_and_return_conditional_losses_741872
input_1'
conv2d_741446:@
conv2d_741448:@+
instance_normalization_741495:@+
instance_normalization_741497:@*
conv2d_1_741518:@�
conv2d_1_741520:	�.
instance_normalization_1_741567:	�.
instance_normalization_1_741569:	�+
conv2d_2_741590:��
conv2d_2_741592:	�.
instance_normalization_2_741639:	�.
instance_normalization_2_741641:	�+
residual_741651:��
residual_741653:	�+
residual_741655:��
residual_741657:	�-
residual_1_741660:�� 
residual_1_741662:	�-
residual_1_741664:�� 
residual_1_741666:	�-
residual_2_741669:�� 
residual_2_741671:	�-
residual_2_741673:�� 
residual_2_741675:	�-
residual_3_741678:�� 
residual_3_741680:	�-
residual_3_741682:�� 
residual_3_741684:	�-
residual_4_741687:�� 
residual_4_741689:	�-
residual_4_741691:�� 
residual_4_741693:	�-
residual_5_741696:�� 
residual_5_741698:	�-
residual_5_741700:�� 
residual_5_741702:	�-
residual_6_741705:�� 
residual_6_741707:	�-
residual_6_741709:�� 
residual_6_741711:	�-
residual_7_741714:�� 
residual_7_741716:	�-
residual_7_741718:�� 
residual_7_741720:	�-
residual_8_741723:�� 
residual_8_741725:	�-
residual_8_741727:�� 
residual_8_741729:	�3
conv2d_transpose_741732:��&
conv2d_transpose_741734:	�.
instance_normalization_3_741781:	�.
instance_normalization_3_741783:	�4
conv2d_transpose_1_741793:@�'
conv2d_transpose_1_741795:@-
instance_normalization_4_741842:@-
instance_normalization_4_741844:@*
conv2d_21_741866:@
conv2d_21_741868:
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�!conv2d_21/StatefulPartitionedCall�(conv2d_transpose/StatefulPartitionedCall�*conv2d_transpose_1/StatefulPartitionedCall�.instance_normalization/StatefulPartitionedCall�0instance_normalization_1/StatefulPartitionedCall�0instance_normalization_2/StatefulPartitionedCall�0instance_normalization_3/StatefulPartitionedCall�0instance_normalization_4/StatefulPartitionedCall� residual/StatefulPartitionedCall�"residual_1/StatefulPartitionedCall�"residual_2/StatefulPartitionedCall�"residual_3/StatefulPartitionedCall�"residual_4/StatefulPartitionedCall�"residual_5/StatefulPartitionedCall�"residual_6/StatefulPartitionedCall�"residual_7/StatefulPartitionedCall�"residual_8/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_741446conv2d_741448*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_741445�
.instance_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0instance_normalization_741495instance_normalization_741497*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *[
fVRT
R__inference_instance_normalization_layer_call_and_return_conditional_losses_741494�
re_lu/PartitionedCallPartitionedCall7instance_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_741505�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_741518conv2d_1_741520*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_741517�
0instance_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0instance_normalization_1_741567instance_normalization_1_741569*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *]
fXRV
T__inference_instance_normalization_1_layer_call_and_return_conditional_losses_741566�
re_lu_1/PartitionedCallPartitionedCall9instance_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *L
fGRE
C__inference_re_lu_1_layer_call_and_return_conditional_losses_741577�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_2_741590conv2d_2_741592*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_741589�
0instance_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0instance_normalization_2_741639instance_normalization_2_741641*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *]
fXRV
T__inference_instance_normalization_2_layer_call_and_return_conditional_losses_741638�
re_lu_2/PartitionedCallPartitionedCall9instance_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_741649�
 residual/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0residual_741651residual_741653residual_741655residual_741657*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_residual_layer_call_and_return_conditional_losses_740717�
"residual_1/StatefulPartitionedCallStatefulPartitionedCall)residual/StatefulPartitionedCall:output:0residual_1_741660residual_1_741662residual_1_741664residual_1_741666*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_1_layer_call_and_return_conditional_losses_740791�
"residual_2/StatefulPartitionedCallStatefulPartitionedCall+residual_1/StatefulPartitionedCall:output:0residual_2_741669residual_2_741671residual_2_741673residual_2_741675*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_2_layer_call_and_return_conditional_losses_740865�
"residual_3/StatefulPartitionedCallStatefulPartitionedCall+residual_2/StatefulPartitionedCall:output:0residual_3_741678residual_3_741680residual_3_741682residual_3_741684*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_3_layer_call_and_return_conditional_losses_740939�
"residual_4/StatefulPartitionedCallStatefulPartitionedCall+residual_3/StatefulPartitionedCall:output:0residual_4_741687residual_4_741689residual_4_741691residual_4_741693*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_4_layer_call_and_return_conditional_losses_741013�
"residual_5/StatefulPartitionedCallStatefulPartitionedCall+residual_4/StatefulPartitionedCall:output:0residual_5_741696residual_5_741698residual_5_741700residual_5_741702*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_5_layer_call_and_return_conditional_losses_741087�
"residual_6/StatefulPartitionedCallStatefulPartitionedCall+residual_5/StatefulPartitionedCall:output:0residual_6_741705residual_6_741707residual_6_741709residual_6_741711*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_6_layer_call_and_return_conditional_losses_741161�
"residual_7/StatefulPartitionedCallStatefulPartitionedCall+residual_6/StatefulPartitionedCall:output:0residual_7_741714residual_7_741716residual_7_741718residual_7_741720*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_7_layer_call_and_return_conditional_losses_741235�
"residual_8/StatefulPartitionedCallStatefulPartitionedCall+residual_7/StatefulPartitionedCall:output:0residual_8_741723residual_8_741725residual_8_741727residual_8_741729*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_8_layer_call_and_return_conditional_losses_741309�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall+residual_8/StatefulPartitionedCall:output:0conv2d_transpose_741732conv2d_transpose_741734*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_741377�
0instance_normalization_3/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0instance_normalization_3_741781instance_normalization_3_741783*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *]
fXRV
T__inference_instance_normalization_3_layer_call_and_return_conditional_losses_741780�
re_lu_3/PartitionedCallPartitionedCall9instance_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_741791�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv2d_transpose_1_741793conv2d_transpose_1_741795*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_741421�
0instance_normalization_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0instance_normalization_4_741842instance_normalization_4_741844*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *]
fXRV
T__inference_instance_normalization_4_layer_call_and_return_conditional_losses_741841�
re_lu_4/PartitionedCallPartitionedCall9instance_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_741852�
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv2d_21_741866conv2d_21_741868*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_741865�
IdentityIdentity*conv2d_21/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall/^instance_normalization/StatefulPartitionedCall1^instance_normalization_1/StatefulPartitionedCall1^instance_normalization_2/StatefulPartitionedCall1^instance_normalization_3/StatefulPartitionedCall1^instance_normalization_4/StatefulPartitionedCall!^residual/StatefulPartitionedCall#^residual_1/StatefulPartitionedCall#^residual_2/StatefulPartitionedCall#^residual_3/StatefulPartitionedCall#^residual_4/StatefulPartitionedCall#^residual_5/StatefulPartitionedCall#^residual_6/StatefulPartitionedCall#^residual_7/StatefulPartitionedCall#^residual_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2`
.instance_normalization/StatefulPartitionedCall.instance_normalization/StatefulPartitionedCall2d
0instance_normalization_1/StatefulPartitionedCall0instance_normalization_1/StatefulPartitionedCall2d
0instance_normalization_2/StatefulPartitionedCall0instance_normalization_2/StatefulPartitionedCall2d
0instance_normalization_3/StatefulPartitionedCall0instance_normalization_3/StatefulPartitionedCall2d
0instance_normalization_4/StatefulPartitionedCall0instance_normalization_4/StatefulPartitionedCall2D
 residual/StatefulPartitionedCall residual/StatefulPartitionedCall2H
"residual_1/StatefulPartitionedCall"residual_1/StatefulPartitionedCall2H
"residual_2/StatefulPartitionedCall"residual_2/StatefulPartitionedCall2H
"residual_3/StatefulPartitionedCall"residual_3/StatefulPartitionedCall2H
"residual_4/StatefulPartitionedCall"residual_4/StatefulPartitionedCall2H
"residual_5/StatefulPartitionedCall"residual_5/StatefulPartitionedCall2H
"residual_6/StatefulPartitionedCall"residual_6/StatefulPartitionedCall2H
"residual_7/StatefulPartitionedCall"residual_7/StatefulPartitionedCall2H
"residual_8/StatefulPartitionedCall"residual_8/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
+__inference_residual_2_layer_call_fn_744401

inputs#
unknown:��
	unknown_0:	�%
	unknown_1:��
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_2_layer_call_and_return_conditional_losses_740865x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
E__inference_conv2d_17_layer_call_and_return_conditional_losses_741210

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������@@�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
*__inference_conv2d_11_layer_call_fn_745005

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_740988x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
E__inference_conv2d_11_layer_call_and_return_conditional_losses_740988

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������@@�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
F__inference_residual_5_layer_call_and_return_conditional_losses_744516

inputsD
(conv2d_13_conv2d_readvariableop_resource:��8
)conv2d_13_biasadd_readvariableop_resource:	�D
(conv2d_14_conv2d_readvariableop_resource:��8
)conv2d_14_biasadd_readvariableop_resource:	�
identity�� conv2d_13/BiasAdd/ReadVariableOp�conv2d_13/Conv2D/ReadVariableOp� conv2d_14/BiasAdd/ReadVariableOp�conv2d_14/Conv2D/ReadVariableOp�
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_13/Conv2DConv2Dinputs'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�m
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_14/Conv2DConv2Dconv2d_13/Relu:activations:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�o
add/addAddV2conv2d_14/BiasAdd:output:0inputs*
T0*0
_output_shapes
:���������@@�Z

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:���������@@�p
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@��
NoOpNoOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
D__inference_conv2d_5_layer_call_and_return_conditional_losses_740766

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������@@�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�

�
B__inference_conv2d_layer_call_and_return_conditional_losses_741445

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:�����������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
*__inference_conv2d_15_layer_call_fn_745083

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_741136x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
D
(__inference_re_lu_2_layer_call_fn_744319

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_741649i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������@@�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������@@�:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
F__inference_residual_7_layer_call_and_return_conditional_losses_741235

inputs,
conv2d_17_741211:��
conv2d_17_741213:	�,
conv2d_18_741227:��
conv2d_18_741229:	�
identity��!conv2d_17/StatefulPartitionedCall�!conv2d_18/StatefulPartitionedCall�
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_17_741211conv2d_17_741213*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_741210�
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0conv2d_18_741227conv2d_18_741229*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_741226
add/addAddV2*conv2d_18/StatefulPartitionedCall:output:0inputs*
T0*0
_output_shapes
:���������@@�Z

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:���������@@�p
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@��
NoOpNoOp"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
+__inference_residual_7_layer_call_fn_744561

inputs#
unknown:��
	unknown_0:	�%
	unknown_1:��
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_7_layer_call_and_return_conditional_losses_741235x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
*__inference_conv2d_12_layer_call_fn_745025

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_741004x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
+__inference_sequential_layer_call_fn_743156

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�&

unknown_25:��

unknown_26:	�&

unknown_27:��

unknown_28:	�&

unknown_29:��

unknown_30:	�&

unknown_31:��

unknown_32:	�&

unknown_33:��

unknown_34:	�&

unknown_35:��

unknown_36:	�&

unknown_37:��

unknown_38:	�&

unknown_39:��

unknown_40:	�&

unknown_41:��

unknown_42:	�&

unknown_43:��

unknown_44:	�&

unknown_45:��

unknown_46:	�&

unknown_47:��

unknown_48:	�

unknown_49:	�

unknown_50:	�%

unknown_51:@�

unknown_52:@

unknown_53:@

unknown_54:@$

unknown_55:@

unknown_56:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_742165y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
_
C__inference_re_lu_4_layer_call_and_return_conditional_losses_741852

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:�����������@d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:�����������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������@:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
*__inference_conv2d_19_layer_call_fn_745161

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_741284x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
_
C__inference_re_lu_3_layer_call_and_return_conditional_losses_741791

inputs
identityQ
ReluReluinputs*
T0*2
_output_shapes 
:������������e
IdentityIdentityRelu:activations:0*
T0*2
_output_shapes 
:������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������:Z V
2
_output_shapes 
:������������
 
_user_specified_nameinputs
�

�
E__inference_conv2d_18_layer_call_and_return_conditional_losses_745152

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
)__inference_conv2d_3_layer_call_fn_744849

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_740692x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
E__inference_conv2d_13_layer_call_and_return_conditional_losses_741062

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������@@�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�

�
E__inference_conv2d_20_layer_call_and_return_conditional_losses_741300

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
E__inference_conv2d_13_layer_call_and_return_conditional_losses_745055

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������@@�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�

�
D__inference_conv2d_6_layer_call_and_return_conditional_losses_744918

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
)__inference_conv2d_1_layer_call_fn_744171

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_741517z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
F__inference_residual_5_layer_call_and_return_conditional_losses_741087

inputs,
conv2d_13_741063:��
conv2d_13_741065:	�,
conv2d_14_741079:��
conv2d_14_741081:	�
identity��!conv2d_13/StatefulPartitionedCall�!conv2d_14/StatefulPartitionedCall�
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_13_741063conv2d_13_741065*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_741062�
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0conv2d_14_741079conv2d_14_741081*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_741078
add/addAddV2*conv2d_14/StatefulPartitionedCall:output:0inputs*
T0*0
_output_shapes
:���������@@�Z

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:���������@@�p
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@��
NoOpNoOp"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
��
�
F__inference_sequential_layer_call_and_return_conditional_losses_742165

inputs'
conv2d_742023:@
conv2d_742025:@+
instance_normalization_742028:@+
instance_normalization_742030:@*
conv2d_1_742034:@�
conv2d_1_742036:	�.
instance_normalization_1_742039:	�.
instance_normalization_1_742041:	�+
conv2d_2_742045:��
conv2d_2_742047:	�.
instance_normalization_2_742050:	�.
instance_normalization_2_742052:	�+
residual_742056:��
residual_742058:	�+
residual_742060:��
residual_742062:	�-
residual_1_742065:�� 
residual_1_742067:	�-
residual_1_742069:�� 
residual_1_742071:	�-
residual_2_742074:�� 
residual_2_742076:	�-
residual_2_742078:�� 
residual_2_742080:	�-
residual_3_742083:�� 
residual_3_742085:	�-
residual_3_742087:�� 
residual_3_742089:	�-
residual_4_742092:�� 
residual_4_742094:	�-
residual_4_742096:�� 
residual_4_742098:	�-
residual_5_742101:�� 
residual_5_742103:	�-
residual_5_742105:�� 
residual_5_742107:	�-
residual_6_742110:�� 
residual_6_742112:	�-
residual_6_742114:�� 
residual_6_742116:	�-
residual_7_742119:�� 
residual_7_742121:	�-
residual_7_742123:�� 
residual_7_742125:	�-
residual_8_742128:�� 
residual_8_742130:	�-
residual_8_742132:�� 
residual_8_742134:	�3
conv2d_transpose_742137:��&
conv2d_transpose_742139:	�.
instance_normalization_3_742142:	�.
instance_normalization_3_742144:	�4
conv2d_transpose_1_742148:@�'
conv2d_transpose_1_742150:@-
instance_normalization_4_742153:@-
instance_normalization_4_742155:@*
conv2d_21_742159:@
conv2d_21_742161:
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�!conv2d_21/StatefulPartitionedCall�(conv2d_transpose/StatefulPartitionedCall�*conv2d_transpose_1/StatefulPartitionedCall�.instance_normalization/StatefulPartitionedCall�0instance_normalization_1/StatefulPartitionedCall�0instance_normalization_2/StatefulPartitionedCall�0instance_normalization_3/StatefulPartitionedCall�0instance_normalization_4/StatefulPartitionedCall� residual/StatefulPartitionedCall�"residual_1/StatefulPartitionedCall�"residual_2/StatefulPartitionedCall�"residual_3/StatefulPartitionedCall�"residual_4/StatefulPartitionedCall�"residual_5/StatefulPartitionedCall�"residual_6/StatefulPartitionedCall�"residual_7/StatefulPartitionedCall�"residual_8/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_742023conv2d_742025*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_741445�
.instance_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0instance_normalization_742028instance_normalization_742030*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *[
fVRT
R__inference_instance_normalization_layer_call_and_return_conditional_losses_741494�
re_lu/PartitionedCallPartitionedCall7instance_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_741505�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_742034conv2d_1_742036*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_741517�
0instance_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0instance_normalization_1_742039instance_normalization_1_742041*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *]
fXRV
T__inference_instance_normalization_1_layer_call_and_return_conditional_losses_741566�
re_lu_1/PartitionedCallPartitionedCall9instance_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *L
fGRE
C__inference_re_lu_1_layer_call_and_return_conditional_losses_741577�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_2_742045conv2d_2_742047*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_741589�
0instance_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0instance_normalization_2_742050instance_normalization_2_742052*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *]
fXRV
T__inference_instance_normalization_2_layer_call_and_return_conditional_losses_741638�
re_lu_2/PartitionedCallPartitionedCall9instance_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_741649�
 residual/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0residual_742056residual_742058residual_742060residual_742062*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_residual_layer_call_and_return_conditional_losses_740717�
"residual_1/StatefulPartitionedCallStatefulPartitionedCall)residual/StatefulPartitionedCall:output:0residual_1_742065residual_1_742067residual_1_742069residual_1_742071*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_1_layer_call_and_return_conditional_losses_740791�
"residual_2/StatefulPartitionedCallStatefulPartitionedCall+residual_1/StatefulPartitionedCall:output:0residual_2_742074residual_2_742076residual_2_742078residual_2_742080*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_2_layer_call_and_return_conditional_losses_740865�
"residual_3/StatefulPartitionedCallStatefulPartitionedCall+residual_2/StatefulPartitionedCall:output:0residual_3_742083residual_3_742085residual_3_742087residual_3_742089*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_3_layer_call_and_return_conditional_losses_740939�
"residual_4/StatefulPartitionedCallStatefulPartitionedCall+residual_3/StatefulPartitionedCall:output:0residual_4_742092residual_4_742094residual_4_742096residual_4_742098*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_4_layer_call_and_return_conditional_losses_741013�
"residual_5/StatefulPartitionedCallStatefulPartitionedCall+residual_4/StatefulPartitionedCall:output:0residual_5_742101residual_5_742103residual_5_742105residual_5_742107*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_5_layer_call_and_return_conditional_losses_741087�
"residual_6/StatefulPartitionedCallStatefulPartitionedCall+residual_5/StatefulPartitionedCall:output:0residual_6_742110residual_6_742112residual_6_742114residual_6_742116*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_6_layer_call_and_return_conditional_losses_741161�
"residual_7/StatefulPartitionedCallStatefulPartitionedCall+residual_6/StatefulPartitionedCall:output:0residual_7_742119residual_7_742121residual_7_742123residual_7_742125*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_7_layer_call_and_return_conditional_losses_741235�
"residual_8/StatefulPartitionedCallStatefulPartitionedCall+residual_7/StatefulPartitionedCall:output:0residual_8_742128residual_8_742130residual_8_742132residual_8_742134*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_8_layer_call_and_return_conditional_losses_741309�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall+residual_8/StatefulPartitionedCall:output:0conv2d_transpose_742137conv2d_transpose_742139*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_741377�
0instance_normalization_3/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0instance_normalization_3_742142instance_normalization_3_742144*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *]
fXRV
T__inference_instance_normalization_3_layer_call_and_return_conditional_losses_741780�
re_lu_3/PartitionedCallPartitionedCall9instance_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_741791�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv2d_transpose_1_742148conv2d_transpose_1_742150*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_741421�
0instance_normalization_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0instance_normalization_4_742153instance_normalization_4_742155*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *]
fXRV
T__inference_instance_normalization_4_layer_call_and_return_conditional_losses_741841�
re_lu_4/PartitionedCallPartitionedCall9instance_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_741852�
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv2d_21_742159conv2d_21_742161*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_741865�
IdentityIdentity*conv2d_21/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall/^instance_normalization/StatefulPartitionedCall1^instance_normalization_1/StatefulPartitionedCall1^instance_normalization_2/StatefulPartitionedCall1^instance_normalization_3/StatefulPartitionedCall1^instance_normalization_4/StatefulPartitionedCall!^residual/StatefulPartitionedCall#^residual_1/StatefulPartitionedCall#^residual_2/StatefulPartitionedCall#^residual_3/StatefulPartitionedCall#^residual_4/StatefulPartitionedCall#^residual_5/StatefulPartitionedCall#^residual_6/StatefulPartitionedCall#^residual_7/StatefulPartitionedCall#^residual_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2`
.instance_normalization/StatefulPartitionedCall.instance_normalization/StatefulPartitionedCall2d
0instance_normalization_1/StatefulPartitionedCall0instance_normalization_1/StatefulPartitionedCall2d
0instance_normalization_2/StatefulPartitionedCall0instance_normalization_2/StatefulPartitionedCall2d
0instance_normalization_3/StatefulPartitionedCall0instance_normalization_3/StatefulPartitionedCall2d
0instance_normalization_4/StatefulPartitionedCall0instance_normalization_4/StatefulPartitionedCall2D
 residual/StatefulPartitionedCall residual/StatefulPartitionedCall2H
"residual_1/StatefulPartitionedCall"residual_1/StatefulPartitionedCall2H
"residual_2/StatefulPartitionedCall"residual_2/StatefulPartitionedCall2H
"residual_3/StatefulPartitionedCall"residual_3/StatefulPartitionedCall2H
"residual_4/StatefulPartitionedCall"residual_4/StatefulPartitionedCall2H
"residual_5/StatefulPartitionedCall"residual_5/StatefulPartitionedCall2H
"residual_6/StatefulPartitionedCall"residual_6/StatefulPartitionedCall2H
"residual_7/StatefulPartitionedCall"residual_7/StatefulPartitionedCall2H
"residual_8/StatefulPartitionedCall"residual_8/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
+__inference_sequential_layer_call_fn_742284
input_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�&

unknown_25:��

unknown_26:	�&

unknown_27:��

unknown_28:	�&

unknown_29:��

unknown_30:	�&

unknown_31:��

unknown_32:	�&

unknown_33:��

unknown_34:	�&

unknown_35:��

unknown_36:	�&

unknown_37:��

unknown_38:	�&

unknown_39:��

unknown_40:	�&

unknown_41:��

unknown_42:	�&

unknown_43:��

unknown_44:	�&

unknown_45:��

unknown_46:	�&

unknown_47:��

unknown_48:	�

unknown_49:	�

unknown_50:	�%

unknown_51:@�

unknown_52:@

unknown_53:@

unknown_54:@$

unknown_55:@

unknown_56:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_742165y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
)__inference_conv2d_9_layer_call_fn_744966

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_740914x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�*
�
T__inference_instance_normalization_3_layer_call_and_return_conditional_losses_741780

inputs.
reshape_readvariableop_resource:	�0
!reshape_1_readvariableop_resource:	�
identity��Reshape/ReadVariableOp�Reshape_1/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(v
moments/StopGradientStopGradientmoments/mean:output:0*
T0*0
_output_shapes
:�����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*2
_output_shapes 
:������������s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(s
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes	
:�*
dtype0f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   |
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*'
_output_shapes
:�w
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:�T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*0
_output_shapes
:����������f
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*0
_output_shapes
:����������v
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*0
_output_shapes
:����������n
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*2
_output_shapes 
:������������{
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*0
_output_shapes
:����������x
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*0
_output_shapes
:����������}
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*2
_output_shapes 
:������������m
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*2
_output_shapes 
:������������z
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":������������: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:Z V
2
_output_shapes 
:������������
 
_user_specified_nameinputs
�

�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_741517

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������j
IdentityIdentityBiasAdd:output:0^NoOp*
T0*2
_output_shapes 
:������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
��
�9
__inference__traced_save_745562
file_prefix>
$read_disablecopyonread_conv2d_kernel:@2
$read_1_disablecopyonread_conv2d_bias:@C
5read_2_disablecopyonread_instance_normalization_gamma:@B
4read_3_disablecopyonread_instance_normalization_beta:@C
(read_4_disablecopyonread_conv2d_1_kernel:@�5
&read_5_disablecopyonread_conv2d_1_bias:	�F
7read_6_disablecopyonread_instance_normalization_1_gamma:	�E
6read_7_disablecopyonread_instance_normalization_1_beta:	�D
(read_8_disablecopyonread_conv2d_2_kernel:��5
&read_9_disablecopyonread_conv2d_2_bias:	�G
8read_10_disablecopyonread_instance_normalization_2_gamma:	�F
7read_11_disablecopyonread_instance_normalization_2_beta:	�M
1read_12_disablecopyonread_conv2d_transpose_kernel:��>
/read_13_disablecopyonread_conv2d_transpose_bias:	�G
8read_14_disablecopyonread_instance_normalization_3_gamma:	�F
7read_15_disablecopyonread_instance_normalization_3_beta:	�N
3read_16_disablecopyonread_conv2d_transpose_1_kernel:@�?
1read_17_disablecopyonread_conv2d_transpose_1_bias:@F
8read_18_disablecopyonread_instance_normalization_4_gamma:@E
7read_19_disablecopyonread_instance_normalization_4_beta:@D
*read_20_disablecopyonread_conv2d_21_kernel:@6
(read_21_disablecopyonread_conv2d_21_bias:N
2read_22_disablecopyonread_residual_conv2d_3_kernel:��?
0read_23_disablecopyonread_residual_conv2d_3_bias:	�N
2read_24_disablecopyonread_residual_conv2d_4_kernel:��?
0read_25_disablecopyonread_residual_conv2d_4_bias:	�P
4read_26_disablecopyonread_residual_1_conv2d_5_kernel:��A
2read_27_disablecopyonread_residual_1_conv2d_5_bias:	�P
4read_28_disablecopyonread_residual_1_conv2d_6_kernel:��A
2read_29_disablecopyonread_residual_1_conv2d_6_bias:	�P
4read_30_disablecopyonread_residual_2_conv2d_7_kernel:��A
2read_31_disablecopyonread_residual_2_conv2d_7_bias:	�P
4read_32_disablecopyonread_residual_2_conv2d_8_kernel:��A
2read_33_disablecopyonread_residual_2_conv2d_8_bias:	�P
4read_34_disablecopyonread_residual_3_conv2d_9_kernel:��A
2read_35_disablecopyonread_residual_3_conv2d_9_bias:	�Q
5read_36_disablecopyonread_residual_3_conv2d_10_kernel:��B
3read_37_disablecopyonread_residual_3_conv2d_10_bias:	�Q
5read_38_disablecopyonread_residual_4_conv2d_11_kernel:��B
3read_39_disablecopyonread_residual_4_conv2d_11_bias:	�Q
5read_40_disablecopyonread_residual_4_conv2d_12_kernel:��B
3read_41_disablecopyonread_residual_4_conv2d_12_bias:	�Q
5read_42_disablecopyonread_residual_5_conv2d_13_kernel:��B
3read_43_disablecopyonread_residual_5_conv2d_13_bias:	�Q
5read_44_disablecopyonread_residual_5_conv2d_14_kernel:��B
3read_45_disablecopyonread_residual_5_conv2d_14_bias:	�Q
5read_46_disablecopyonread_residual_6_conv2d_15_kernel:��B
3read_47_disablecopyonread_residual_6_conv2d_15_bias:	�Q
5read_48_disablecopyonread_residual_6_conv2d_16_kernel:��B
3read_49_disablecopyonread_residual_6_conv2d_16_bias:	�Q
5read_50_disablecopyonread_residual_7_conv2d_17_kernel:��B
3read_51_disablecopyonread_residual_7_conv2d_17_bias:	�Q
5read_52_disablecopyonread_residual_7_conv2d_18_kernel:��B
3read_53_disablecopyonread_residual_7_conv2d_18_bias:	�Q
5read_54_disablecopyonread_residual_8_conv2d_19_kernel:��B
3read_55_disablecopyonread_residual_8_conv2d_19_bias:	�Q
5read_56_disablecopyonread_residual_8_conv2d_20_kernel:��B
3read_57_disablecopyonread_residual_8_conv2d_20_bias:	�
savev2_const
identity_117��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: v
Read/DisableCopyOnReadDisableCopyOnRead$read_disablecopyonread_conv2d_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp$read_disablecopyonread_conv2d_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:@x
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_conv2d_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_conv2d_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_2/DisableCopyOnReadDisableCopyOnRead5read_2_disablecopyonread_instance_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp5read_2_disablecopyonread_instance_normalization_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_3/DisableCopyOnReadDisableCopyOnRead4read_3_disablecopyonread_instance_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp4read_3_disablecopyonread_instance_normalization_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:@|
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_conv2d_1_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0v

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�l

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*'
_output_shapes
:@�z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_conv2d_1_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_conv2d_1_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_6/DisableCopyOnReadDisableCopyOnRead7read_6_disablecopyonread_instance_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp7read_6_disablecopyonread_instance_normalization_1_gamma^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_7/DisableCopyOnReadDisableCopyOnRead6read_7_disablecopyonread_instance_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp6read_7_disablecopyonread_instance_normalization_1_beta^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_8/DisableCopyOnReadDisableCopyOnRead(read_8_disablecopyonread_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp(read_8_disablecopyonread_conv2d_2_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0x
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*(
_output_shapes
:��z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_conv2d_2_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_conv2d_2_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_10/DisableCopyOnReadDisableCopyOnRead8read_10_disablecopyonread_instance_normalization_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp8read_10_disablecopyonread_instance_normalization_2_gamma^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_11/DisableCopyOnReadDisableCopyOnRead7read_11_disablecopyonread_instance_normalization_2_beta"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp7read_11_disablecopyonread_instance_normalization_2_beta^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_12/DisableCopyOnReadDisableCopyOnRead1read_12_disablecopyonread_conv2d_transpose_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp1read_12_disablecopyonread_conv2d_transpose_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_13/DisableCopyOnReadDisableCopyOnRead/read_13_disablecopyonread_conv2d_transpose_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp/read_13_disablecopyonread_conv2d_transpose_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_14/DisableCopyOnReadDisableCopyOnRead8read_14_disablecopyonread_instance_normalization_3_gamma"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp8read_14_disablecopyonread_instance_normalization_3_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_15/DisableCopyOnReadDisableCopyOnRead7read_15_disablecopyonread_instance_normalization_3_beta"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp7read_15_disablecopyonread_instance_normalization_3_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_16/DisableCopyOnReadDisableCopyOnRead3read_16_disablecopyonread_conv2d_transpose_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp3read_16_disablecopyonread_conv2d_transpose_1_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_17/DisableCopyOnReadDisableCopyOnRead1read_17_disablecopyonread_conv2d_transpose_1_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp1read_17_disablecopyonread_conv2d_transpose_1_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_18/DisableCopyOnReadDisableCopyOnRead8read_18_disablecopyonread_instance_normalization_4_gamma"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp8read_18_disablecopyonread_instance_normalization_4_gamma^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_19/DisableCopyOnReadDisableCopyOnRead7read_19_disablecopyonread_instance_normalization_4_beta"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp7read_19_disablecopyonread_instance_normalization_4_beta^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_20/DisableCopyOnReadDisableCopyOnRead*read_20_disablecopyonread_conv2d_21_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp*read_20_disablecopyonread_conv2d_21_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*&
_output_shapes
:@}
Read_21/DisableCopyOnReadDisableCopyOnRead(read_21_disablecopyonread_conv2d_21_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp(read_21_disablecopyonread_conv2d_21_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_22/DisableCopyOnReadDisableCopyOnRead2read_22_disablecopyonread_residual_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp2read_22_disablecopyonread_residual_conv2d_3_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_23/DisableCopyOnReadDisableCopyOnRead0read_23_disablecopyonread_residual_conv2d_3_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp0read_23_disablecopyonread_residual_conv2d_3_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_24/DisableCopyOnReadDisableCopyOnRead2read_24_disablecopyonread_residual_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp2read_24_disablecopyonread_residual_conv2d_4_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_25/DisableCopyOnReadDisableCopyOnRead0read_25_disablecopyonread_residual_conv2d_4_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp0read_25_disablecopyonread_residual_conv2d_4_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_26/DisableCopyOnReadDisableCopyOnRead4read_26_disablecopyonread_residual_1_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp4read_26_disablecopyonread_residual_1_conv2d_5_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_27/DisableCopyOnReadDisableCopyOnRead2read_27_disablecopyonread_residual_1_conv2d_5_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp2read_27_disablecopyonread_residual_1_conv2d_5_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_28/DisableCopyOnReadDisableCopyOnRead4read_28_disablecopyonread_residual_1_conv2d_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp4read_28_disablecopyonread_residual_1_conv2d_6_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_29/DisableCopyOnReadDisableCopyOnRead2read_29_disablecopyonread_residual_1_conv2d_6_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp2read_29_disablecopyonread_residual_1_conv2d_6_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_30/DisableCopyOnReadDisableCopyOnRead4read_30_disablecopyonread_residual_2_conv2d_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp4read_30_disablecopyonread_residual_2_conv2d_7_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_31/DisableCopyOnReadDisableCopyOnRead2read_31_disablecopyonread_residual_2_conv2d_7_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp2read_31_disablecopyonread_residual_2_conv2d_7_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_32/DisableCopyOnReadDisableCopyOnRead4read_32_disablecopyonread_residual_2_conv2d_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp4read_32_disablecopyonread_residual_2_conv2d_8_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_33/DisableCopyOnReadDisableCopyOnRead2read_33_disablecopyonread_residual_2_conv2d_8_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp2read_33_disablecopyonread_residual_2_conv2d_8_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_34/DisableCopyOnReadDisableCopyOnRead4read_34_disablecopyonread_residual_3_conv2d_9_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp4read_34_disablecopyonread_residual_3_conv2d_9_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_35/DisableCopyOnReadDisableCopyOnRead2read_35_disablecopyonread_residual_3_conv2d_9_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp2read_35_disablecopyonread_residual_3_conv2d_9_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_36/DisableCopyOnReadDisableCopyOnRead5read_36_disablecopyonread_residual_3_conv2d_10_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp5read_36_disablecopyonread_residual_3_conv2d_10_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_37/DisableCopyOnReadDisableCopyOnRead3read_37_disablecopyonread_residual_3_conv2d_10_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp3read_37_disablecopyonread_residual_3_conv2d_10_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_38/DisableCopyOnReadDisableCopyOnRead5read_38_disablecopyonread_residual_4_conv2d_11_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp5read_38_disablecopyonread_residual_4_conv2d_11_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_39/DisableCopyOnReadDisableCopyOnRead3read_39_disablecopyonread_residual_4_conv2d_11_bias"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp3read_39_disablecopyonread_residual_4_conv2d_11_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_40/DisableCopyOnReadDisableCopyOnRead5read_40_disablecopyonread_residual_4_conv2d_12_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp5read_40_disablecopyonread_residual_4_conv2d_12_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_41/DisableCopyOnReadDisableCopyOnRead3read_41_disablecopyonread_residual_4_conv2d_12_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp3read_41_disablecopyonread_residual_4_conv2d_12_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_42/DisableCopyOnReadDisableCopyOnRead5read_42_disablecopyonread_residual_5_conv2d_13_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp5read_42_disablecopyonread_residual_5_conv2d_13_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_43/DisableCopyOnReadDisableCopyOnRead3read_43_disablecopyonread_residual_5_conv2d_13_bias"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp3read_43_disablecopyonread_residual_5_conv2d_13_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_44/DisableCopyOnReadDisableCopyOnRead5read_44_disablecopyonread_residual_5_conv2d_14_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp5read_44_disablecopyonread_residual_5_conv2d_14_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_45/DisableCopyOnReadDisableCopyOnRead3read_45_disablecopyonread_residual_5_conv2d_14_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp3read_45_disablecopyonread_residual_5_conv2d_14_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_46/DisableCopyOnReadDisableCopyOnRead5read_46_disablecopyonread_residual_6_conv2d_15_kernel"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp5read_46_disablecopyonread_residual_6_conv2d_15_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_47/DisableCopyOnReadDisableCopyOnRead3read_47_disablecopyonread_residual_6_conv2d_15_bias"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp3read_47_disablecopyonread_residual_6_conv2d_15_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_48/DisableCopyOnReadDisableCopyOnRead5read_48_disablecopyonread_residual_6_conv2d_16_kernel"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp5read_48_disablecopyonread_residual_6_conv2d_16_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_49/DisableCopyOnReadDisableCopyOnRead3read_49_disablecopyonread_residual_6_conv2d_16_bias"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp3read_49_disablecopyonread_residual_6_conv2d_16_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_50/DisableCopyOnReadDisableCopyOnRead5read_50_disablecopyonread_residual_7_conv2d_17_kernel"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp5read_50_disablecopyonread_residual_7_conv2d_17_kernel^Read_50/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_51/DisableCopyOnReadDisableCopyOnRead3read_51_disablecopyonread_residual_7_conv2d_17_bias"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp3read_51_disablecopyonread_residual_7_conv2d_17_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_52/DisableCopyOnReadDisableCopyOnRead5read_52_disablecopyonread_residual_7_conv2d_18_kernel"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp5read_52_disablecopyonread_residual_7_conv2d_18_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_53/DisableCopyOnReadDisableCopyOnRead3read_53_disablecopyonread_residual_7_conv2d_18_bias"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp3read_53_disablecopyonread_residual_7_conv2d_18_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_54/DisableCopyOnReadDisableCopyOnRead5read_54_disablecopyonread_residual_8_conv2d_19_kernel"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp5read_54_disablecopyonread_residual_8_conv2d_19_kernel^Read_54/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_55/DisableCopyOnReadDisableCopyOnRead3read_55_disablecopyonread_residual_8_conv2d_19_bias"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp3read_55_disablecopyonread_residual_8_conv2d_19_bias^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_56/DisableCopyOnReadDisableCopyOnRead5read_56_disablecopyonread_residual_8_conv2d_20_kernel"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp5read_56_disablecopyonread_residual_8_conv2d_20_kernel^Read_56/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_57/DisableCopyOnReadDisableCopyOnRead3read_57_disablecopyonread_residual_8_conv2d_20_bias"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp3read_57_disablecopyonread_residual_8_conv2d_20_bias^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*�
value�B�;B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*�
value�B~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *I
dtypes?
=2;�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_116Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_117IdentityIdentity_116:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "%
identity_117Identity_117:output:0*�
_input_shapesz
x: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:;

_output_shapes
: 
�

�
D__inference_conv2d_8_layer_call_and_return_conditional_losses_744957

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
]
A__inference_re_lu_layer_call_and_return_conditional_losses_744162

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:�����������@d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:�����������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������@:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
E__inference_conv2d_15_layer_call_and_return_conditional_losses_741136

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������@@�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
+__inference_sequential_layer_call_fn_743277

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�&

unknown_25:��

unknown_26:	�&

unknown_27:��

unknown_28:	�&

unknown_29:��

unknown_30:	�&

unknown_31:��

unknown_32:	�&

unknown_33:��

unknown_34:	�&

unknown_35:��

unknown_36:	�&

unknown_37:��

unknown_38:	�&

unknown_39:��

unknown_40:	�&

unknown_41:��

unknown_42:	�&

unknown_43:��

unknown_44:	�&

unknown_45:��

unknown_46:	�&

unknown_47:��

unknown_48:	�

unknown_49:	�

unknown_50:	�%

unknown_51:@�

unknown_52:@

unknown_53:@

unknown_54:@$

unknown_55:@

unknown_56:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_742431y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
��
�
F__inference_sequential_layer_call_and_return_conditional_losses_742017
input_1'
conv2d_741875:@
conv2d_741877:@+
instance_normalization_741880:@+
instance_normalization_741882:@*
conv2d_1_741886:@�
conv2d_1_741888:	�.
instance_normalization_1_741891:	�.
instance_normalization_1_741893:	�+
conv2d_2_741897:��
conv2d_2_741899:	�.
instance_normalization_2_741902:	�.
instance_normalization_2_741904:	�+
residual_741908:��
residual_741910:	�+
residual_741912:��
residual_741914:	�-
residual_1_741917:�� 
residual_1_741919:	�-
residual_1_741921:�� 
residual_1_741923:	�-
residual_2_741926:�� 
residual_2_741928:	�-
residual_2_741930:�� 
residual_2_741932:	�-
residual_3_741935:�� 
residual_3_741937:	�-
residual_3_741939:�� 
residual_3_741941:	�-
residual_4_741944:�� 
residual_4_741946:	�-
residual_4_741948:�� 
residual_4_741950:	�-
residual_5_741953:�� 
residual_5_741955:	�-
residual_5_741957:�� 
residual_5_741959:	�-
residual_6_741962:�� 
residual_6_741964:	�-
residual_6_741966:�� 
residual_6_741968:	�-
residual_7_741971:�� 
residual_7_741973:	�-
residual_7_741975:�� 
residual_7_741977:	�-
residual_8_741980:�� 
residual_8_741982:	�-
residual_8_741984:�� 
residual_8_741986:	�3
conv2d_transpose_741989:��&
conv2d_transpose_741991:	�.
instance_normalization_3_741994:	�.
instance_normalization_3_741996:	�4
conv2d_transpose_1_742000:@�'
conv2d_transpose_1_742002:@-
instance_normalization_4_742005:@-
instance_normalization_4_742007:@*
conv2d_21_742011:@
conv2d_21_742013:
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�!conv2d_21/StatefulPartitionedCall�(conv2d_transpose/StatefulPartitionedCall�*conv2d_transpose_1/StatefulPartitionedCall�.instance_normalization/StatefulPartitionedCall�0instance_normalization_1/StatefulPartitionedCall�0instance_normalization_2/StatefulPartitionedCall�0instance_normalization_3/StatefulPartitionedCall�0instance_normalization_4/StatefulPartitionedCall� residual/StatefulPartitionedCall�"residual_1/StatefulPartitionedCall�"residual_2/StatefulPartitionedCall�"residual_3/StatefulPartitionedCall�"residual_4/StatefulPartitionedCall�"residual_5/StatefulPartitionedCall�"residual_6/StatefulPartitionedCall�"residual_7/StatefulPartitionedCall�"residual_8/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_741875conv2d_741877*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_741445�
.instance_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0instance_normalization_741880instance_normalization_741882*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *[
fVRT
R__inference_instance_normalization_layer_call_and_return_conditional_losses_741494�
re_lu/PartitionedCallPartitionedCall7instance_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_741505�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_741886conv2d_1_741888*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_741517�
0instance_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0instance_normalization_1_741891instance_normalization_1_741893*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *]
fXRV
T__inference_instance_normalization_1_layer_call_and_return_conditional_losses_741566�
re_lu_1/PartitionedCallPartitionedCall9instance_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *L
fGRE
C__inference_re_lu_1_layer_call_and_return_conditional_losses_741577�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_2_741897conv2d_2_741899*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_741589�
0instance_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0instance_normalization_2_741902instance_normalization_2_741904*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *]
fXRV
T__inference_instance_normalization_2_layer_call_and_return_conditional_losses_741638�
re_lu_2/PartitionedCallPartitionedCall9instance_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_741649�
 residual/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0residual_741908residual_741910residual_741912residual_741914*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_residual_layer_call_and_return_conditional_losses_740717�
"residual_1/StatefulPartitionedCallStatefulPartitionedCall)residual/StatefulPartitionedCall:output:0residual_1_741917residual_1_741919residual_1_741921residual_1_741923*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_1_layer_call_and_return_conditional_losses_740791�
"residual_2/StatefulPartitionedCallStatefulPartitionedCall+residual_1/StatefulPartitionedCall:output:0residual_2_741926residual_2_741928residual_2_741930residual_2_741932*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_2_layer_call_and_return_conditional_losses_740865�
"residual_3/StatefulPartitionedCallStatefulPartitionedCall+residual_2/StatefulPartitionedCall:output:0residual_3_741935residual_3_741937residual_3_741939residual_3_741941*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_3_layer_call_and_return_conditional_losses_740939�
"residual_4/StatefulPartitionedCallStatefulPartitionedCall+residual_3/StatefulPartitionedCall:output:0residual_4_741944residual_4_741946residual_4_741948residual_4_741950*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_4_layer_call_and_return_conditional_losses_741013�
"residual_5/StatefulPartitionedCallStatefulPartitionedCall+residual_4/StatefulPartitionedCall:output:0residual_5_741953residual_5_741955residual_5_741957residual_5_741959*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_5_layer_call_and_return_conditional_losses_741087�
"residual_6/StatefulPartitionedCallStatefulPartitionedCall+residual_5/StatefulPartitionedCall:output:0residual_6_741962residual_6_741964residual_6_741966residual_6_741968*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_6_layer_call_and_return_conditional_losses_741161�
"residual_7/StatefulPartitionedCallStatefulPartitionedCall+residual_6/StatefulPartitionedCall:output:0residual_7_741971residual_7_741973residual_7_741975residual_7_741977*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_7_layer_call_and_return_conditional_losses_741235�
"residual_8/StatefulPartitionedCallStatefulPartitionedCall+residual_7/StatefulPartitionedCall:output:0residual_8_741980residual_8_741982residual_8_741984residual_8_741986*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_8_layer_call_and_return_conditional_losses_741309�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall+residual_8/StatefulPartitionedCall:output:0conv2d_transpose_741989conv2d_transpose_741991*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_741377�
0instance_normalization_3/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0instance_normalization_3_741994instance_normalization_3_741996*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *]
fXRV
T__inference_instance_normalization_3_layer_call_and_return_conditional_losses_741780�
re_lu_3/PartitionedCallPartitionedCall9instance_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_741791�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv2d_transpose_1_742000conv2d_transpose_1_742002*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_741421�
0instance_normalization_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0instance_normalization_4_742005instance_normalization_4_742007*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *]
fXRV
T__inference_instance_normalization_4_layer_call_and_return_conditional_losses_741841�
re_lu_4/PartitionedCallPartitionedCall9instance_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_741852�
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv2d_21_742011conv2d_21_742013*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_741865�
IdentityIdentity*conv2d_21/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall/^instance_normalization/StatefulPartitionedCall1^instance_normalization_1/StatefulPartitionedCall1^instance_normalization_2/StatefulPartitionedCall1^instance_normalization_3/StatefulPartitionedCall1^instance_normalization_4/StatefulPartitionedCall!^residual/StatefulPartitionedCall#^residual_1/StatefulPartitionedCall#^residual_2/StatefulPartitionedCall#^residual_3/StatefulPartitionedCall#^residual_4/StatefulPartitionedCall#^residual_5/StatefulPartitionedCall#^residual_6/StatefulPartitionedCall#^residual_7/StatefulPartitionedCall#^residual_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2`
.instance_normalization/StatefulPartitionedCall.instance_normalization/StatefulPartitionedCall2d
0instance_normalization_1/StatefulPartitionedCall0instance_normalization_1/StatefulPartitionedCall2d
0instance_normalization_2/StatefulPartitionedCall0instance_normalization_2/StatefulPartitionedCall2d
0instance_normalization_3/StatefulPartitionedCall0instance_normalization_3/StatefulPartitionedCall2d
0instance_normalization_4/StatefulPartitionedCall0instance_normalization_4/StatefulPartitionedCall2D
 residual/StatefulPartitionedCall residual/StatefulPartitionedCall2H
"residual_1/StatefulPartitionedCall"residual_1/StatefulPartitionedCall2H
"residual_2/StatefulPartitionedCall"residual_2/StatefulPartitionedCall2H
"residual_3/StatefulPartitionedCall"residual_3/StatefulPartitionedCall2H
"residual_4/StatefulPartitionedCall"residual_4/StatefulPartitionedCall2H
"residual_5/StatefulPartitionedCall"residual_5/StatefulPartitionedCall2H
"residual_6/StatefulPartitionedCall"residual_6/StatefulPartitionedCall2H
"residual_7/StatefulPartitionedCall"residual_7/StatefulPartitionedCall2H
"residual_8/StatefulPartitionedCall"residual_8/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
B
&__inference_re_lu_layer_call_fn_744157

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_741505j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:�����������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������@:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
F__inference_residual_2_layer_call_and_return_conditional_losses_744420

inputsC
'conv2d_7_conv2d_readvariableop_resource:��7
(conv2d_7_biasadd_readvariableop_resource:	�C
'conv2d_8_conv2d_readvariableop_resource:��7
(conv2d_8_biasadd_readvariableop_resource:	�
identity��conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�conv2d_8/BiasAdd/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_7/Conv2DConv2Dinputs&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�k
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_8/Conv2DConv2Dconv2d_7/Relu:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�n
add/addAddV2conv2d_8/BiasAdd:output:0inputs*
T0*0
_output_shapes
:���������@@�Z

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:���������@@�p
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@��
NoOpNoOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
F__inference_residual_6_layer_call_and_return_conditional_losses_744548

inputsD
(conv2d_15_conv2d_readvariableop_resource:��8
)conv2d_15_biasadd_readvariableop_resource:	�D
(conv2d_16_conv2d_readvariableop_resource:��8
)conv2d_16_biasadd_readvariableop_resource:	�
identity�� conv2d_15/BiasAdd/ReadVariableOp�conv2d_15/Conv2D/ReadVariableOp� conv2d_16/BiasAdd/ReadVariableOp�conv2d_16/Conv2D/ReadVariableOp�
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_15/Conv2DConv2Dinputs'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�m
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_16/Conv2DConv2Dconv2d_15/Relu:activations:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�o
add/addAddV2conv2d_16/BiasAdd:output:0inputs*
T0*0
_output_shapes
:���������@@�Z

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:���������@@�p
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@��
NoOpNoOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
+__inference_residual_6_layer_call_fn_744529

inputs#
unknown:��
	unknown_0:	�%
	unknown_1:��
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_6_layer_call_and_return_conditional_losses_741161x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
9__inference_instance_normalization_3_layer_call_fn_744663

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *]
fXRV
T__inference_instance_normalization_3_layer_call_and_return_conditional_losses_741780z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:������������
 
_user_specified_nameinputs
� 
�
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_744758

inputsC
(conv2d_transpose_readvariableop_resource:@�-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_19_layer_call_and_return_conditional_losses_741284

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������@@�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
E__inference_conv2d_19_layer_call_and_return_conditional_losses_745172

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������@@�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
D__inference_conv2d_9_layer_call_and_return_conditional_losses_744977

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������@@�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
F__inference_residual_3_layer_call_and_return_conditional_losses_744452

inputsC
'conv2d_9_conv2d_readvariableop_resource:��7
(conv2d_9_biasadd_readvariableop_resource:	�D
(conv2d_10_conv2d_readvariableop_resource:��8
)conv2d_10_biasadd_readvariableop_resource:	�
identity�� conv2d_10/BiasAdd/ReadVariableOp�conv2d_10/Conv2D/ReadVariableOp�conv2d_9/BiasAdd/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOp�
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_9/Conv2DConv2Dinputs&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�k
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_10/Conv2DConv2Dconv2d_9/Relu:activations:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�o
add/addAddV2conv2d_10/BiasAdd:output:0inputs*
T0*0
_output_shapes
:���������@@�Z

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:���������@@�p
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@��
NoOpNoOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�

�
E__inference_conv2d_18_layer_call_and_return_conditional_losses_741226

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
E__inference_conv2d_21_layer_call_and_return_conditional_losses_741865

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������Z
TanhTanhBiasAdd:output:0*
T0*1
_output_shapes
:�����������a
IdentityIdentityTanh:y:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
+__inference_residual_3_layer_call_fn_744433

inputs#
unknown:��
	unknown_0:	�%
	unknown_1:��
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_3_layer_call_and_return_conditional_losses_740939x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
F__inference_residual_3_layer_call_and_return_conditional_losses_740939

inputs+
conv2d_9_740915:��
conv2d_9_740917:	�,
conv2d_10_740931:��
conv2d_10_740933:	�
identity��!conv2d_10/StatefulPartitionedCall� conv2d_9/StatefulPartitionedCall�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_740915conv2d_9_740917*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_740914�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_740931conv2d_10_740933*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_740930
add/addAddV2*conv2d_10/StatefulPartitionedCall:output:0inputs*
T0*0
_output_shapes
:���������@@�Z

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:���������@@�p
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@��
NoOpNoOp"^conv2d_10/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�

�
E__inference_conv2d_20_layer_call_and_return_conditional_losses_745191

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
F__inference_residual_6_layer_call_and_return_conditional_losses_741161

inputs,
conv2d_15_741137:��
conv2d_15_741139:	�,
conv2d_16_741153:��
conv2d_16_741155:	�
identity��!conv2d_15/StatefulPartitionedCall�!conv2d_16/StatefulPartitionedCall�
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_15_741137conv2d_15_741139*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_741136�
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0conv2d_16_741153conv2d_16_741155*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_741152
add/addAddV2*conv2d_16/StatefulPartitionedCall:output:0inputs*
T0*0
_output_shapes
:���������@@�Z

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:���������@@�p
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@��
NoOpNoOp"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
D__inference_residual_layer_call_and_return_conditional_losses_744356

inputsC
'conv2d_3_conv2d_readvariableop_resource:��7
(conv2d_3_biasadd_readvariableop_resource:	�C
'conv2d_4_conv2d_readvariableop_resource:��7
(conv2d_4_biasadd_readvariableop_resource:	�
identity��conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�k
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_4/Conv2DConv2Dconv2d_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�n
add/addAddV2conv2d_4/BiasAdd:output:0inputs*
T0*0
_output_shapes
:���������@@�Z

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:���������@@�p
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@��
NoOpNoOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�

�
E__inference_conv2d_12_layer_call_and_return_conditional_losses_745035

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�

�
E__inference_conv2d_14_layer_call_and_return_conditional_losses_745074

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�

�
E__inference_conv2d_14_layer_call_and_return_conditional_losses_741078

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
)__inference_conv2d_6_layer_call_fn_744908

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_740782x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
F__inference_residual_1_layer_call_and_return_conditional_losses_740791

inputs+
conv2d_5_740767:��
conv2d_5_740769:	�+
conv2d_6_740783:��
conv2d_6_740785:	�
identity�� conv2d_5/StatefulPartitionedCall� conv2d_6/StatefulPartitionedCall�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_740767conv2d_5_740769*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_740766�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_740783conv2d_6_740785*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_740782~
add/addAddV2)conv2d_6/StatefulPartitionedCall:output:0inputs*
T0*0
_output_shapes
:���������@@�Z

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:���������@@�p
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@��
NoOpNoOp!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�*
�
T__inference_instance_normalization_4_layer_call_and_return_conditional_losses_741841

inputs-
reshape_readvariableop_resource:@/
!reshape_1_readvariableop_resource:@
identity��Reshape/ReadVariableOp�Reshape_1/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������@*
	keep_dims(u
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:���������@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*1
_output_shapes
:�����������@s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:���������@*
	keep_dims(r
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:@*
dtype0f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   {
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:@v
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:@*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:@T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:���������@e
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:���������@u
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:���������@m
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*1
_output_shapes
:�����������@z
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:���������@w
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:���������@|
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*1
_output_shapes
:�����������@l
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*1
_output_shapes
:�����������@z
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
_
C__inference_re_lu_3_layer_call_and_return_conditional_losses_744716

inputs
identityQ
ReluReluinputs*
T0*2
_output_shapes 
:������������e
IdentityIdentityRelu:activations:0*
T0*2
_output_shapes 
:������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������:Z V
2
_output_shapes 
:������������
 
_user_specified_nameinputs
�
�
F__inference_residual_7_layer_call_and_return_conditional_losses_744580

inputsD
(conv2d_17_conv2d_readvariableop_resource:��8
)conv2d_17_biasadd_readvariableop_resource:	�D
(conv2d_18_conv2d_readvariableop_resource:��8
)conv2d_18_biasadd_readvariableop_resource:	�
identity�� conv2d_17/BiasAdd/ReadVariableOp�conv2d_17/Conv2D/ReadVariableOp� conv2d_18/BiasAdd/ReadVariableOp�conv2d_18/Conv2D/ReadVariableOp�
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_17/Conv2DConv2Dinputs'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�m
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_18/Conv2DConv2Dconv2d_17/Relu:activations:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�o
add/addAddV2conv2d_18/BiasAdd:output:0inputs*
T0*0
_output_shapes
:���������@@�Z

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:���������@@�p
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@��
NoOpNoOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
F__inference_residual_4_layer_call_and_return_conditional_losses_741013

inputs,
conv2d_11_740989:��
conv2d_11_740991:	�,
conv2d_12_741005:��
conv2d_12_741007:	�
identity��!conv2d_11/StatefulPartitionedCall�!conv2d_12/StatefulPartitionedCall�
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_11_740989conv2d_11_740991*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_740988�
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0conv2d_12_741005conv2d_12_741007*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_741004
add/addAddV2*conv2d_12/StatefulPartitionedCall:output:0inputs*
T0*0
_output_shapes
:���������@@�Z

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:���������@@�p
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@��
NoOpNoOp"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
]
A__inference_re_lu_layer_call_and_return_conditional_losses_741505

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:�����������@d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:�����������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������@:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�

�
E__inference_conv2d_16_layer_call_and_return_conditional_losses_741152

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
'__inference_conv2d_layer_call_fn_744090

inputs!
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_741445y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
D__inference_conv2d_6_layer_call_and_return_conditional_losses_740782

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
*__inference_conv2d_20_layer_call_fn_745181

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_741300x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
� 
�
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_744654

inputsD
(conv2d_transpose_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
*__inference_conv2d_10_layer_call_fn_744986

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_740930x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
9__inference_instance_normalization_4_layer_call_fn_744767

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *]
fXRV
T__inference_instance_normalization_4_layer_call_and_return_conditional_losses_741841y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
+__inference_residual_8_layer_call_fn_744593

inputs#
unknown:��
	unknown_0:	�%
	unknown_1:��
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_8_layer_call_and_return_conditional_losses_741309x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
*__inference_conv2d_13_layer_call_fn_745044

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_741062x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
*__inference_conv2d_16_layer_call_fn_745103

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_741152x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
��
�5
F__inference_sequential_layer_call_and_return_conditional_losses_743679

inputs?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@D
6instance_normalization_reshape_readvariableop_resource:@F
8instance_normalization_reshape_1_readvariableop_resource:@B
'conv2d_1_conv2d_readvariableop_resource:@�7
(conv2d_1_biasadd_readvariableop_resource:	�G
8instance_normalization_1_reshape_readvariableop_resource:	�I
:instance_normalization_1_reshape_1_readvariableop_resource:	�C
'conv2d_2_conv2d_readvariableop_resource:��7
(conv2d_2_biasadd_readvariableop_resource:	�G
8instance_normalization_2_reshape_readvariableop_resource:	�I
:instance_normalization_2_reshape_1_readvariableop_resource:	�L
0residual_conv2d_3_conv2d_readvariableop_resource:��@
1residual_conv2d_3_biasadd_readvariableop_resource:	�L
0residual_conv2d_4_conv2d_readvariableop_resource:��@
1residual_conv2d_4_biasadd_readvariableop_resource:	�N
2residual_1_conv2d_5_conv2d_readvariableop_resource:��B
3residual_1_conv2d_5_biasadd_readvariableop_resource:	�N
2residual_1_conv2d_6_conv2d_readvariableop_resource:��B
3residual_1_conv2d_6_biasadd_readvariableop_resource:	�N
2residual_2_conv2d_7_conv2d_readvariableop_resource:��B
3residual_2_conv2d_7_biasadd_readvariableop_resource:	�N
2residual_2_conv2d_8_conv2d_readvariableop_resource:��B
3residual_2_conv2d_8_biasadd_readvariableop_resource:	�N
2residual_3_conv2d_9_conv2d_readvariableop_resource:��B
3residual_3_conv2d_9_biasadd_readvariableop_resource:	�O
3residual_3_conv2d_10_conv2d_readvariableop_resource:��C
4residual_3_conv2d_10_biasadd_readvariableop_resource:	�O
3residual_4_conv2d_11_conv2d_readvariableop_resource:��C
4residual_4_conv2d_11_biasadd_readvariableop_resource:	�O
3residual_4_conv2d_12_conv2d_readvariableop_resource:��C
4residual_4_conv2d_12_biasadd_readvariableop_resource:	�O
3residual_5_conv2d_13_conv2d_readvariableop_resource:��C
4residual_5_conv2d_13_biasadd_readvariableop_resource:	�O
3residual_5_conv2d_14_conv2d_readvariableop_resource:��C
4residual_5_conv2d_14_biasadd_readvariableop_resource:	�O
3residual_6_conv2d_15_conv2d_readvariableop_resource:��C
4residual_6_conv2d_15_biasadd_readvariableop_resource:	�O
3residual_6_conv2d_16_conv2d_readvariableop_resource:��C
4residual_6_conv2d_16_biasadd_readvariableop_resource:	�O
3residual_7_conv2d_17_conv2d_readvariableop_resource:��C
4residual_7_conv2d_17_biasadd_readvariableop_resource:	�O
3residual_7_conv2d_18_conv2d_readvariableop_resource:��C
4residual_7_conv2d_18_biasadd_readvariableop_resource:	�O
3residual_8_conv2d_19_conv2d_readvariableop_resource:��C
4residual_8_conv2d_19_biasadd_readvariableop_resource:	�O
3residual_8_conv2d_20_conv2d_readvariableop_resource:��C
4residual_8_conv2d_20_biasadd_readvariableop_resource:	�U
9conv2d_transpose_conv2d_transpose_readvariableop_resource:��?
0conv2d_transpose_biasadd_readvariableop_resource:	�G
8instance_normalization_3_reshape_readvariableop_resource:	�I
:instance_normalization_3_reshape_1_readvariableop_resource:	�V
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@�@
2conv2d_transpose_1_biasadd_readvariableop_resource:@F
8instance_normalization_4_reshape_readvariableop_resource:@H
:instance_normalization_4_reshape_1_readvariableop_resource:@B
(conv2d_21_conv2d_readvariableop_resource:@7
)conv2d_21_biasadd_readvariableop_resource:
identity��conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp� conv2d_21/BiasAdd/ReadVariableOp�conv2d_21/Conv2D/ReadVariableOp�'conv2d_transpose/BiasAdd/ReadVariableOp�0conv2d_transpose/conv2d_transpose/ReadVariableOp�)conv2d_transpose_1/BiasAdd/ReadVariableOp�2conv2d_transpose_1/conv2d_transpose/ReadVariableOp�-instance_normalization/Reshape/ReadVariableOp�/instance_normalization/Reshape_1/ReadVariableOp�/instance_normalization_1/Reshape/ReadVariableOp�1instance_normalization_1/Reshape_1/ReadVariableOp�/instance_normalization_2/Reshape/ReadVariableOp�1instance_normalization_2/Reshape_1/ReadVariableOp�/instance_normalization_3/Reshape/ReadVariableOp�1instance_normalization_3/Reshape_1/ReadVariableOp�/instance_normalization_4/Reshape/ReadVariableOp�1instance_normalization_4/Reshape_1/ReadVariableOp�(residual/conv2d_3/BiasAdd/ReadVariableOp�'residual/conv2d_3/Conv2D/ReadVariableOp�(residual/conv2d_4/BiasAdd/ReadVariableOp�'residual/conv2d_4/Conv2D/ReadVariableOp�*residual_1/conv2d_5/BiasAdd/ReadVariableOp�)residual_1/conv2d_5/Conv2D/ReadVariableOp�*residual_1/conv2d_6/BiasAdd/ReadVariableOp�)residual_1/conv2d_6/Conv2D/ReadVariableOp�*residual_2/conv2d_7/BiasAdd/ReadVariableOp�)residual_2/conv2d_7/Conv2D/ReadVariableOp�*residual_2/conv2d_8/BiasAdd/ReadVariableOp�)residual_2/conv2d_8/Conv2D/ReadVariableOp�+residual_3/conv2d_10/BiasAdd/ReadVariableOp�*residual_3/conv2d_10/Conv2D/ReadVariableOp�*residual_3/conv2d_9/BiasAdd/ReadVariableOp�)residual_3/conv2d_9/Conv2D/ReadVariableOp�+residual_4/conv2d_11/BiasAdd/ReadVariableOp�*residual_4/conv2d_11/Conv2D/ReadVariableOp�+residual_4/conv2d_12/BiasAdd/ReadVariableOp�*residual_4/conv2d_12/Conv2D/ReadVariableOp�+residual_5/conv2d_13/BiasAdd/ReadVariableOp�*residual_5/conv2d_13/Conv2D/ReadVariableOp�+residual_5/conv2d_14/BiasAdd/ReadVariableOp�*residual_5/conv2d_14/Conv2D/ReadVariableOp�+residual_6/conv2d_15/BiasAdd/ReadVariableOp�*residual_6/conv2d_15/Conv2D/ReadVariableOp�+residual_6/conv2d_16/BiasAdd/ReadVariableOp�*residual_6/conv2d_16/Conv2D/ReadVariableOp�+residual_7/conv2d_17/BiasAdd/ReadVariableOp�*residual_7/conv2d_17/Conv2D/ReadVariableOp�+residual_7/conv2d_18/BiasAdd/ReadVariableOp�*residual_7/conv2d_18/Conv2D/ReadVariableOp�+residual_8/conv2d_19/BiasAdd/ReadVariableOp�*residual_8/conv2d_19/Conv2D/ReadVariableOp�+residual_8/conv2d_20/BiasAdd/ReadVariableOp�*residual_8/conv2d_20/Conv2D/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@q
instance_normalization/ShapeShapeconv2d/BiasAdd:output:0*
T0*
_output_shapes
::��t
*instance_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,instance_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,instance_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$instance_normalization/strided_sliceStridedSlice%instance_normalization/Shape:output:03instance_normalization/strided_slice/stack:output:05instance_normalization/strided_slice/stack_1:output:05instance_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
,instance_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.instance_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.instance_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&instance_normalization/strided_slice_1StridedSlice%instance_normalization/Shape:output:05instance_normalization/strided_slice_1/stack:output:07instance_normalization/strided_slice_1/stack_1:output:07instance_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
,instance_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.instance_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.instance_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&instance_normalization/strided_slice_2StridedSlice%instance_normalization/Shape:output:05instance_normalization/strided_slice_2/stack:output:07instance_normalization/strided_slice_2/stack_1:output:07instance_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
,instance_normalization/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.instance_normalization/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.instance_normalization/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&instance_normalization/strided_slice_3StridedSlice%instance_normalization/Shape:output:05instance_normalization/strided_slice_3/stack:output:07instance_normalization/strided_slice_3/stack_1:output:07instance_normalization/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
5instance_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
#instance_normalization/moments/meanMeanconv2d/BiasAdd:output:0>instance_normalization/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������@*
	keep_dims(�
+instance_normalization/moments/StopGradientStopGradient,instance_normalization/moments/mean:output:0*
T0*/
_output_shapes
:���������@�
0instance_normalization/moments/SquaredDifferenceSquaredDifferenceconv2d/BiasAdd:output:04instance_normalization/moments/StopGradient:output:0*
T0*1
_output_shapes
:�����������@�
9instance_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
'instance_normalization/moments/varianceMean4instance_normalization/moments/SquaredDifference:z:0Binstance_normalization/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:���������@*
	keep_dims(�
-instance_normalization/Reshape/ReadVariableOpReadVariableOp6instance_normalization_reshape_readvariableop_resource*
_output_shapes
:@*
dtype0}
$instance_normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
instance_normalization/ReshapeReshape5instance_normalization/Reshape/ReadVariableOp:value:0-instance_normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:@�
/instance_normalization/Reshape_1/ReadVariableOpReadVariableOp8instance_normalization_reshape_1_readvariableop_resource*
_output_shapes
:@*
dtype0
&instance_normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
 instance_normalization/Reshape_1Reshape7instance_normalization/Reshape_1/ReadVariableOp:value:0/instance_normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:@k
&instance_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$instance_normalization/batchnorm/addAddV20instance_normalization/moments/variance:output:0/instance_normalization/batchnorm/add/y:output:0*
T0*/
_output_shapes
:���������@�
&instance_normalization/batchnorm/RsqrtRsqrt(instance_normalization/batchnorm/add:z:0*
T0*/
_output_shapes
:���������@�
$instance_normalization/batchnorm/mulMul*instance_normalization/batchnorm/Rsqrt:y:0'instance_normalization/Reshape:output:0*
T0*/
_output_shapes
:���������@�
&instance_normalization/batchnorm/mul_1Mulconv2d/BiasAdd:output:0(instance_normalization/batchnorm/mul:z:0*
T0*1
_output_shapes
:�����������@�
&instance_normalization/batchnorm/mul_2Mul,instance_normalization/moments/mean:output:0(instance_normalization/batchnorm/mul:z:0*
T0*/
_output_shapes
:���������@�
$instance_normalization/batchnorm/subSub)instance_normalization/Reshape_1:output:0*instance_normalization/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:���������@�
&instance_normalization/batchnorm/add_1AddV2*instance_normalization/batchnorm/mul_1:z:0(instance_normalization/batchnorm/sub:z:0*
T0*1
_output_shapes
:�����������@z

re_lu/ReluRelu*instance_normalization/batchnorm/add_1:z:0*
T0*1
_output_shapes
:�����������@�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_1/Conv2DConv2Dre_lu/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������*
paddingSAME*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������u
instance_normalization_1/ShapeShapeconv2d_1/BiasAdd:output:0*
T0*
_output_shapes
::��v
,instance_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.instance_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.instance_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&instance_normalization_1/strided_sliceStridedSlice'instance_normalization_1/Shape:output:05instance_normalization_1/strided_slice/stack:output:07instance_normalization_1/strided_slice/stack_1:output:07instance_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_1/strided_slice_1StridedSlice'instance_normalization_1/Shape:output:07instance_normalization_1/strided_slice_1/stack:output:09instance_normalization_1/strided_slice_1/stack_1:output:09instance_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_1/strided_slice_2StridedSlice'instance_normalization_1/Shape:output:07instance_normalization_1/strided_slice_2/stack:output:09instance_normalization_1/strided_slice_2/stack_1:output:09instance_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_1/strided_slice_3StridedSlice'instance_normalization_1/Shape:output:07instance_normalization_1/strided_slice_3/stack:output:09instance_normalization_1/strided_slice_3/stack_1:output:09instance_normalization_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
7instance_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
%instance_normalization_1/moments/meanMeanconv2d_1/BiasAdd:output:0@instance_normalization_1/moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
-instance_normalization_1/moments/StopGradientStopGradient.instance_normalization_1/moments/mean:output:0*
T0*0
_output_shapes
:�����������
2instance_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv2d_1/BiasAdd:output:06instance_normalization_1/moments/StopGradient:output:0*
T0*2
_output_shapes 
:�������������
;instance_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
)instance_normalization_1/moments/varianceMean6instance_normalization_1/moments/SquaredDifference:z:0Dinstance_normalization_1/moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
/instance_normalization_1/Reshape/ReadVariableOpReadVariableOp8instance_normalization_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0
&instance_normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
 instance_normalization_1/ReshapeReshape7instance_normalization_1/Reshape/ReadVariableOp:value:0/instance_normalization_1/Reshape/shape:output:0*
T0*'
_output_shapes
:��
1instance_normalization_1/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_1_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(instance_normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
"instance_normalization_1/Reshape_1Reshape9instance_normalization_1/Reshape_1/ReadVariableOp:value:01instance_normalization_1/Reshape_1/shape:output:0*
T0*'
_output_shapes
:�m
(instance_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&instance_normalization_1/batchnorm/addAddV22instance_normalization_1/moments/variance:output:01instance_normalization_1/batchnorm/add/y:output:0*
T0*0
_output_shapes
:�����������
(instance_normalization_1/batchnorm/RsqrtRsqrt*instance_normalization_1/batchnorm/add:z:0*
T0*0
_output_shapes
:�����������
&instance_normalization_1/batchnorm/mulMul,instance_normalization_1/batchnorm/Rsqrt:y:0)instance_normalization_1/Reshape:output:0*
T0*0
_output_shapes
:�����������
(instance_normalization_1/batchnorm/mul_1Mulconv2d_1/BiasAdd:output:0*instance_normalization_1/batchnorm/mul:z:0*
T0*2
_output_shapes 
:�������������
(instance_normalization_1/batchnorm/mul_2Mul.instance_normalization_1/moments/mean:output:0*instance_normalization_1/batchnorm/mul:z:0*
T0*0
_output_shapes
:�����������
&instance_normalization_1/batchnorm/subSub+instance_normalization_1/Reshape_1:output:0,instance_normalization_1/batchnorm/mul_2:z:0*
T0*0
_output_shapes
:�����������
(instance_normalization_1/batchnorm/add_1AddV2,instance_normalization_1/batchnorm/mul_1:z:0*instance_normalization_1/batchnorm/sub:z:0*
T0*2
_output_shapes 
:������������
re_lu_1/ReluRelu,instance_normalization_1/batchnorm/add_1:z:0*
T0*2
_output_shapes 
:�������������
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_2/Conv2DConv2Dre_lu_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�u
instance_normalization_2/ShapeShapeconv2d_2/BiasAdd:output:0*
T0*
_output_shapes
::��v
,instance_normalization_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.instance_normalization_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.instance_normalization_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&instance_normalization_2/strided_sliceStridedSlice'instance_normalization_2/Shape:output:05instance_normalization_2/strided_slice/stack:output:07instance_normalization_2/strided_slice/stack_1:output:07instance_normalization_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_2/strided_slice_1StridedSlice'instance_normalization_2/Shape:output:07instance_normalization_2/strided_slice_1/stack:output:09instance_normalization_2/strided_slice_1/stack_1:output:09instance_normalization_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_2/strided_slice_2StridedSlice'instance_normalization_2/Shape:output:07instance_normalization_2/strided_slice_2/stack:output:09instance_normalization_2/strided_slice_2/stack_1:output:09instance_normalization_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_2/strided_slice_3StridedSlice'instance_normalization_2/Shape:output:07instance_normalization_2/strided_slice_3/stack:output:09instance_normalization_2/strided_slice_3/stack_1:output:09instance_normalization_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
7instance_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
%instance_normalization_2/moments/meanMeanconv2d_2/BiasAdd:output:0@instance_normalization_2/moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
-instance_normalization_2/moments/StopGradientStopGradient.instance_normalization_2/moments/mean:output:0*
T0*0
_output_shapes
:�����������
2instance_normalization_2/moments/SquaredDifferenceSquaredDifferenceconv2d_2/BiasAdd:output:06instance_normalization_2/moments/StopGradient:output:0*
T0*0
_output_shapes
:���������@@��
;instance_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
)instance_normalization_2/moments/varianceMean6instance_normalization_2/moments/SquaredDifference:z:0Dinstance_normalization_2/moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
/instance_normalization_2/Reshape/ReadVariableOpReadVariableOp8instance_normalization_2_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0
&instance_normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
 instance_normalization_2/ReshapeReshape7instance_normalization_2/Reshape/ReadVariableOp:value:0/instance_normalization_2/Reshape/shape:output:0*
T0*'
_output_shapes
:��
1instance_normalization_2/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_2_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(instance_normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
"instance_normalization_2/Reshape_1Reshape9instance_normalization_2/Reshape_1/ReadVariableOp:value:01instance_normalization_2/Reshape_1/shape:output:0*
T0*'
_output_shapes
:�m
(instance_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&instance_normalization_2/batchnorm/addAddV22instance_normalization_2/moments/variance:output:01instance_normalization_2/batchnorm/add/y:output:0*
T0*0
_output_shapes
:�����������
(instance_normalization_2/batchnorm/RsqrtRsqrt*instance_normalization_2/batchnorm/add:z:0*
T0*0
_output_shapes
:�����������
&instance_normalization_2/batchnorm/mulMul,instance_normalization_2/batchnorm/Rsqrt:y:0)instance_normalization_2/Reshape:output:0*
T0*0
_output_shapes
:�����������
(instance_normalization_2/batchnorm/mul_1Mulconv2d_2/BiasAdd:output:0*instance_normalization_2/batchnorm/mul:z:0*
T0*0
_output_shapes
:���������@@��
(instance_normalization_2/batchnorm/mul_2Mul.instance_normalization_2/moments/mean:output:0*instance_normalization_2/batchnorm/mul:z:0*
T0*0
_output_shapes
:�����������
&instance_normalization_2/batchnorm/subSub+instance_normalization_2/Reshape_1:output:0,instance_normalization_2/batchnorm/mul_2:z:0*
T0*0
_output_shapes
:�����������
(instance_normalization_2/batchnorm/add_1AddV2,instance_normalization_2/batchnorm/mul_1:z:0*instance_normalization_2/batchnorm/sub:z:0*
T0*0
_output_shapes
:���������@@�}
re_lu_2/ReluRelu,instance_normalization_2/batchnorm/add_1:z:0*
T0*0
_output_shapes
:���������@@��
'residual/conv2d_3/Conv2D/ReadVariableOpReadVariableOp0residual_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual/conv2d_3/Conv2DConv2Dre_lu_2/Relu:activations:0/residual/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
(residual/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp1residual_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual/conv2d_3/BiasAddBiasAdd!residual/conv2d_3/Conv2D:output:00residual/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�}
residual/conv2d_3/ReluRelu"residual/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
'residual/conv2d_4/Conv2D/ReadVariableOpReadVariableOp0residual_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual/conv2d_4/Conv2DConv2D$residual/conv2d_3/Relu:activations:0/residual/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
(residual/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp1residual_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual/conv2d_4/BiasAddBiasAdd!residual/conv2d_4/Conv2D:output:00residual/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual/add/addAddV2"residual/conv2d_4/BiasAdd:output:0re_lu_2/Relu:activations:0*
T0*0
_output_shapes
:���������@@�l
residual/re_lu/ReluReluresidual/add/add:z:0*
T0*0
_output_shapes
:���������@@��
)residual_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp2residual_1_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_1/conv2d_5/Conv2DConv2D!residual/re_lu/Relu:activations:01residual_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
*residual_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp3residual_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_1/conv2d_5/BiasAddBiasAdd#residual_1/conv2d_5/Conv2D:output:02residual_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_1/conv2d_5/ReluRelu$residual_1/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
)residual_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp2residual_1_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_1/conv2d_6/Conv2DConv2D&residual_1/conv2d_5/Relu:activations:01residual_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
*residual_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp3residual_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_1/conv2d_6/BiasAddBiasAdd#residual_1/conv2d_6/Conv2D:output:02residual_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_1/add_1/addAddV2$residual_1/conv2d_6/BiasAdd:output:0!residual/re_lu/Relu:activations:0*
T0*0
_output_shapes
:���������@@�t
residual_1/re_lu_1/ReluReluresidual_1/add_1/add:z:0*
T0*0
_output_shapes
:���������@@��
)residual_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp2residual_2_conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_2/conv2d_7/Conv2DConv2D%residual_1/re_lu_1/Relu:activations:01residual_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
*residual_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp3residual_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_2/conv2d_7/BiasAddBiasAdd#residual_2/conv2d_7/Conv2D:output:02residual_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_2/conv2d_7/ReluRelu$residual_2/conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
)residual_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp2residual_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_2/conv2d_8/Conv2DConv2D&residual_2/conv2d_7/Relu:activations:01residual_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
*residual_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp3residual_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_2/conv2d_8/BiasAddBiasAdd#residual_2/conv2d_8/Conv2D:output:02residual_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_2/add_2/addAddV2$residual_2/conv2d_8/BiasAdd:output:0%residual_1/re_lu_1/Relu:activations:0*
T0*0
_output_shapes
:���������@@�t
residual_2/re_lu_2/ReluReluresidual_2/add_2/add:z:0*
T0*0
_output_shapes
:���������@@��
)residual_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp2residual_3_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_3/conv2d_9/Conv2DConv2D%residual_2/re_lu_2/Relu:activations:01residual_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
*residual_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp3residual_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_3/conv2d_9/BiasAddBiasAdd#residual_3/conv2d_9/Conv2D:output:02residual_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_3/conv2d_9/ReluRelu$residual_3/conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
*residual_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp3residual_3_conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_3/conv2d_10/Conv2DConv2D&residual_3/conv2d_9/Relu:activations:02residual_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp4residual_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_3/conv2d_10/BiasAddBiasAdd$residual_3/conv2d_10/Conv2D:output:03residual_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_3/add_3/addAddV2%residual_3/conv2d_10/BiasAdd:output:0%residual_2/re_lu_2/Relu:activations:0*
T0*0
_output_shapes
:���������@@�t
residual_3/re_lu_3/ReluReluresidual_3/add_3/add:z:0*
T0*0
_output_shapes
:���������@@��
*residual_4/conv2d_11/Conv2D/ReadVariableOpReadVariableOp3residual_4_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_4/conv2d_11/Conv2DConv2D%residual_3/re_lu_3/Relu:activations:02residual_4/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_4/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp4residual_4_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_4/conv2d_11/BiasAddBiasAdd$residual_4/conv2d_11/Conv2D:output:03residual_4/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_4/conv2d_11/ReluRelu%residual_4/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
*residual_4/conv2d_12/Conv2D/ReadVariableOpReadVariableOp3residual_4_conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_4/conv2d_12/Conv2DConv2D'residual_4/conv2d_11/Relu:activations:02residual_4/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_4/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp4residual_4_conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_4/conv2d_12/BiasAddBiasAdd$residual_4/conv2d_12/Conv2D:output:03residual_4/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_4/add_4/addAddV2%residual_4/conv2d_12/BiasAdd:output:0%residual_3/re_lu_3/Relu:activations:0*
T0*0
_output_shapes
:���������@@�t
residual_4/re_lu_4/ReluReluresidual_4/add_4/add:z:0*
T0*0
_output_shapes
:���������@@��
*residual_5/conv2d_13/Conv2D/ReadVariableOpReadVariableOp3residual_5_conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_5/conv2d_13/Conv2DConv2D%residual_4/re_lu_4/Relu:activations:02residual_5/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_5/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp4residual_5_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_5/conv2d_13/BiasAddBiasAdd$residual_5/conv2d_13/Conv2D:output:03residual_5/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_5/conv2d_13/ReluRelu%residual_5/conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
*residual_5/conv2d_14/Conv2D/ReadVariableOpReadVariableOp3residual_5_conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_5/conv2d_14/Conv2DConv2D'residual_5/conv2d_13/Relu:activations:02residual_5/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_5/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp4residual_5_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_5/conv2d_14/BiasAddBiasAdd$residual_5/conv2d_14/Conv2D:output:03residual_5/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_5/add_5/addAddV2%residual_5/conv2d_14/BiasAdd:output:0%residual_4/re_lu_4/Relu:activations:0*
T0*0
_output_shapes
:���������@@�t
residual_5/re_lu_5/ReluReluresidual_5/add_5/add:z:0*
T0*0
_output_shapes
:���������@@��
*residual_6/conv2d_15/Conv2D/ReadVariableOpReadVariableOp3residual_6_conv2d_15_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_6/conv2d_15/Conv2DConv2D%residual_5/re_lu_5/Relu:activations:02residual_6/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_6/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp4residual_6_conv2d_15_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_6/conv2d_15/BiasAddBiasAdd$residual_6/conv2d_15/Conv2D:output:03residual_6/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_6/conv2d_15/ReluRelu%residual_6/conv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
*residual_6/conv2d_16/Conv2D/ReadVariableOpReadVariableOp3residual_6_conv2d_16_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_6/conv2d_16/Conv2DConv2D'residual_6/conv2d_15/Relu:activations:02residual_6/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_6/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp4residual_6_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_6/conv2d_16/BiasAddBiasAdd$residual_6/conv2d_16/Conv2D:output:03residual_6/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_6/add_6/addAddV2%residual_6/conv2d_16/BiasAdd:output:0%residual_5/re_lu_5/Relu:activations:0*
T0*0
_output_shapes
:���������@@�t
residual_6/re_lu_6/ReluReluresidual_6/add_6/add:z:0*
T0*0
_output_shapes
:���������@@��
*residual_7/conv2d_17/Conv2D/ReadVariableOpReadVariableOp3residual_7_conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_7/conv2d_17/Conv2DConv2D%residual_6/re_lu_6/Relu:activations:02residual_7/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_7/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp4residual_7_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_7/conv2d_17/BiasAddBiasAdd$residual_7/conv2d_17/Conv2D:output:03residual_7/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_7/conv2d_17/ReluRelu%residual_7/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
*residual_7/conv2d_18/Conv2D/ReadVariableOpReadVariableOp3residual_7_conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_7/conv2d_18/Conv2DConv2D'residual_7/conv2d_17/Relu:activations:02residual_7/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_7/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp4residual_7_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_7/conv2d_18/BiasAddBiasAdd$residual_7/conv2d_18/Conv2D:output:03residual_7/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_7/add_7/addAddV2%residual_7/conv2d_18/BiasAdd:output:0%residual_6/re_lu_6/Relu:activations:0*
T0*0
_output_shapes
:���������@@�t
residual_7/re_lu_7/ReluReluresidual_7/add_7/add:z:0*
T0*0
_output_shapes
:���������@@��
*residual_8/conv2d_19/Conv2D/ReadVariableOpReadVariableOp3residual_8_conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_8/conv2d_19/Conv2DConv2D%residual_7/re_lu_7/Relu:activations:02residual_8/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_8/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp4residual_8_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_8/conv2d_19/BiasAddBiasAdd$residual_8/conv2d_19/Conv2D:output:03residual_8/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_8/conv2d_19/ReluRelu%residual_8/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
*residual_8/conv2d_20/Conv2D/ReadVariableOpReadVariableOp3residual_8_conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_8/conv2d_20/Conv2DConv2D'residual_8/conv2d_19/Relu:activations:02residual_8/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
+residual_8/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp4residual_8_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
residual_8/conv2d_20/BiasAddBiasAdd$residual_8/conv2d_20/Conv2D:output:03residual_8/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
residual_8/add_8/addAddV2%residual_8/conv2d_20/BiasAdd:output:0%residual_7/re_lu_7/Relu:activations:0*
T0*0
_output_shapes
:���������@@�t
residual_8/re_lu_8/ReluReluresidual_8/add_8/add:z:0*
T0*0
_output_shapes
:���������@@�y
conv2d_transpose/ShapeShape%residual_8/re_lu_8/Relu:activations:0*
T0*
_output_shapes
::��n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�[
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�[
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%residual_8/re_lu_8/Relu:activations:0*
T0*2
_output_shapes 
:������������*
paddingSAME*
strides
�
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������}
instance_normalization_3/ShapeShape!conv2d_transpose/BiasAdd:output:0*
T0*
_output_shapes
::��v
,instance_normalization_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.instance_normalization_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.instance_normalization_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&instance_normalization_3/strided_sliceStridedSlice'instance_normalization_3/Shape:output:05instance_normalization_3/strided_slice/stack:output:07instance_normalization_3/strided_slice/stack_1:output:07instance_normalization_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_3/strided_slice_1StridedSlice'instance_normalization_3/Shape:output:07instance_normalization_3/strided_slice_1/stack:output:09instance_normalization_3/strided_slice_1/stack_1:output:09instance_normalization_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_3/strided_slice_2StridedSlice'instance_normalization_3/Shape:output:07instance_normalization_3/strided_slice_2/stack:output:09instance_normalization_3/strided_slice_2/stack_1:output:09instance_normalization_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_3/strided_slice_3StridedSlice'instance_normalization_3/Shape:output:07instance_normalization_3/strided_slice_3/stack:output:09instance_normalization_3/strided_slice_3/stack_1:output:09instance_normalization_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
7instance_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
%instance_normalization_3/moments/meanMean!conv2d_transpose/BiasAdd:output:0@instance_normalization_3/moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
-instance_normalization_3/moments/StopGradientStopGradient.instance_normalization_3/moments/mean:output:0*
T0*0
_output_shapes
:�����������
2instance_normalization_3/moments/SquaredDifferenceSquaredDifference!conv2d_transpose/BiasAdd:output:06instance_normalization_3/moments/StopGradient:output:0*
T0*2
_output_shapes 
:�������������
;instance_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
)instance_normalization_3/moments/varianceMean6instance_normalization_3/moments/SquaredDifference:z:0Dinstance_normalization_3/moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
/instance_normalization_3/Reshape/ReadVariableOpReadVariableOp8instance_normalization_3_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0
&instance_normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
 instance_normalization_3/ReshapeReshape7instance_normalization_3/Reshape/ReadVariableOp:value:0/instance_normalization_3/Reshape/shape:output:0*
T0*'
_output_shapes
:��
1instance_normalization_3/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_3_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(instance_normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
"instance_normalization_3/Reshape_1Reshape9instance_normalization_3/Reshape_1/ReadVariableOp:value:01instance_normalization_3/Reshape_1/shape:output:0*
T0*'
_output_shapes
:�m
(instance_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&instance_normalization_3/batchnorm/addAddV22instance_normalization_3/moments/variance:output:01instance_normalization_3/batchnorm/add/y:output:0*
T0*0
_output_shapes
:�����������
(instance_normalization_3/batchnorm/RsqrtRsqrt*instance_normalization_3/batchnorm/add:z:0*
T0*0
_output_shapes
:�����������
&instance_normalization_3/batchnorm/mulMul,instance_normalization_3/batchnorm/Rsqrt:y:0)instance_normalization_3/Reshape:output:0*
T0*0
_output_shapes
:�����������
(instance_normalization_3/batchnorm/mul_1Mul!conv2d_transpose/BiasAdd:output:0*instance_normalization_3/batchnorm/mul:z:0*
T0*2
_output_shapes 
:�������������
(instance_normalization_3/batchnorm/mul_2Mul.instance_normalization_3/moments/mean:output:0*instance_normalization_3/batchnorm/mul:z:0*
T0*0
_output_shapes
:�����������
&instance_normalization_3/batchnorm/subSub+instance_normalization_3/Reshape_1:output:0,instance_normalization_3/batchnorm/mul_2:z:0*
T0*0
_output_shapes
:�����������
(instance_normalization_3/batchnorm/add_1AddV2,instance_normalization_3/batchnorm/mul_1:z:0*instance_normalization_3/batchnorm/sub:z:0*
T0*2
_output_shapes 
:������������
re_lu_3/ReluRelu,instance_normalization_3/batchnorm/add_1:z:0*
T0*2
_output_shapes 
:������������p
conv2d_transpose_1/ShapeShapere_lu_3/Relu:activations:0*
T0*
_output_shapes
::��p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�]
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0re_lu_3/Relu:activations:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@
instance_normalization_4/ShapeShape#conv2d_transpose_1/BiasAdd:output:0*
T0*
_output_shapes
::��v
,instance_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.instance_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.instance_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&instance_normalization_4/strided_sliceStridedSlice'instance_normalization_4/Shape:output:05instance_normalization_4/strided_slice/stack:output:07instance_normalization_4/strided_slice/stack_1:output:07instance_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_4/strided_slice_1StridedSlice'instance_normalization_4/Shape:output:07instance_normalization_4/strided_slice_1/stack:output:09instance_normalization_4/strided_slice_1/stack_1:output:09instance_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_4/strided_slice_2StridedSlice'instance_normalization_4/Shape:output:07instance_normalization_4/strided_slice_2/stack:output:09instance_normalization_4/strided_slice_2/stack_1:output:09instance_normalization_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.instance_normalization_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0instance_normalization_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(instance_normalization_4/strided_slice_3StridedSlice'instance_normalization_4/Shape:output:07instance_normalization_4/strided_slice_3/stack:output:09instance_normalization_4/strided_slice_3/stack_1:output:09instance_normalization_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
7instance_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
%instance_normalization_4/moments/meanMean#conv2d_transpose_1/BiasAdd:output:0@instance_normalization_4/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������@*
	keep_dims(�
-instance_normalization_4/moments/StopGradientStopGradient.instance_normalization_4/moments/mean:output:0*
T0*/
_output_shapes
:���������@�
2instance_normalization_4/moments/SquaredDifferenceSquaredDifference#conv2d_transpose_1/BiasAdd:output:06instance_normalization_4/moments/StopGradient:output:0*
T0*1
_output_shapes
:�����������@�
;instance_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
)instance_normalization_4/moments/varianceMean6instance_normalization_4/moments/SquaredDifference:z:0Dinstance_normalization_4/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:���������@*
	keep_dims(�
/instance_normalization_4/Reshape/ReadVariableOpReadVariableOp8instance_normalization_4_reshape_readvariableop_resource*
_output_shapes
:@*
dtype0
&instance_normalization_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
 instance_normalization_4/ReshapeReshape7instance_normalization_4/Reshape/ReadVariableOp:value:0/instance_normalization_4/Reshape/shape:output:0*
T0*&
_output_shapes
:@�
1instance_normalization_4/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_4_reshape_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
(instance_normalization_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
"instance_normalization_4/Reshape_1Reshape9instance_normalization_4/Reshape_1/ReadVariableOp:value:01instance_normalization_4/Reshape_1/shape:output:0*
T0*&
_output_shapes
:@m
(instance_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&instance_normalization_4/batchnorm/addAddV22instance_normalization_4/moments/variance:output:01instance_normalization_4/batchnorm/add/y:output:0*
T0*/
_output_shapes
:���������@�
(instance_normalization_4/batchnorm/RsqrtRsqrt*instance_normalization_4/batchnorm/add:z:0*
T0*/
_output_shapes
:���������@�
&instance_normalization_4/batchnorm/mulMul,instance_normalization_4/batchnorm/Rsqrt:y:0)instance_normalization_4/Reshape:output:0*
T0*/
_output_shapes
:���������@�
(instance_normalization_4/batchnorm/mul_1Mul#conv2d_transpose_1/BiasAdd:output:0*instance_normalization_4/batchnorm/mul:z:0*
T0*1
_output_shapes
:�����������@�
(instance_normalization_4/batchnorm/mul_2Mul.instance_normalization_4/moments/mean:output:0*instance_normalization_4/batchnorm/mul:z:0*
T0*/
_output_shapes
:���������@�
&instance_normalization_4/batchnorm/subSub+instance_normalization_4/Reshape_1:output:0,instance_normalization_4/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:���������@�
(instance_normalization_4/batchnorm/add_1AddV2,instance_normalization_4/batchnorm/mul_1:z:0*instance_normalization_4/batchnorm/sub:z:0*
T0*1
_output_shapes
:�����������@~
re_lu_4/ReluRelu,instance_normalization_4/batchnorm/add_1:z:0*
T0*1
_output_shapes
:�����������@�
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d_21/Conv2DConv2Dre_lu_4/Relu:activations:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������n
conv2d_21/TanhTanhconv2d_21/BiasAdd:output:0*
T0*1
_output_shapes
:�����������k
IdentityIdentityconv2d_21/Tanh:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp.^instance_normalization/Reshape/ReadVariableOp0^instance_normalization/Reshape_1/ReadVariableOp0^instance_normalization_1/Reshape/ReadVariableOp2^instance_normalization_1/Reshape_1/ReadVariableOp0^instance_normalization_2/Reshape/ReadVariableOp2^instance_normalization_2/Reshape_1/ReadVariableOp0^instance_normalization_3/Reshape/ReadVariableOp2^instance_normalization_3/Reshape_1/ReadVariableOp0^instance_normalization_4/Reshape/ReadVariableOp2^instance_normalization_4/Reshape_1/ReadVariableOp)^residual/conv2d_3/BiasAdd/ReadVariableOp(^residual/conv2d_3/Conv2D/ReadVariableOp)^residual/conv2d_4/BiasAdd/ReadVariableOp(^residual/conv2d_4/Conv2D/ReadVariableOp+^residual_1/conv2d_5/BiasAdd/ReadVariableOp*^residual_1/conv2d_5/Conv2D/ReadVariableOp+^residual_1/conv2d_6/BiasAdd/ReadVariableOp*^residual_1/conv2d_6/Conv2D/ReadVariableOp+^residual_2/conv2d_7/BiasAdd/ReadVariableOp*^residual_2/conv2d_7/Conv2D/ReadVariableOp+^residual_2/conv2d_8/BiasAdd/ReadVariableOp*^residual_2/conv2d_8/Conv2D/ReadVariableOp,^residual_3/conv2d_10/BiasAdd/ReadVariableOp+^residual_3/conv2d_10/Conv2D/ReadVariableOp+^residual_3/conv2d_9/BiasAdd/ReadVariableOp*^residual_3/conv2d_9/Conv2D/ReadVariableOp,^residual_4/conv2d_11/BiasAdd/ReadVariableOp+^residual_4/conv2d_11/Conv2D/ReadVariableOp,^residual_4/conv2d_12/BiasAdd/ReadVariableOp+^residual_4/conv2d_12/Conv2D/ReadVariableOp,^residual_5/conv2d_13/BiasAdd/ReadVariableOp+^residual_5/conv2d_13/Conv2D/ReadVariableOp,^residual_5/conv2d_14/BiasAdd/ReadVariableOp+^residual_5/conv2d_14/Conv2D/ReadVariableOp,^residual_6/conv2d_15/BiasAdd/ReadVariableOp+^residual_6/conv2d_15/Conv2D/ReadVariableOp,^residual_6/conv2d_16/BiasAdd/ReadVariableOp+^residual_6/conv2d_16/Conv2D/ReadVariableOp,^residual_7/conv2d_17/BiasAdd/ReadVariableOp+^residual_7/conv2d_17/Conv2D/ReadVariableOp,^residual_7/conv2d_18/BiasAdd/ReadVariableOp+^residual_7/conv2d_18/Conv2D/ReadVariableOp,^residual_8/conv2d_19/BiasAdd/ReadVariableOp+^residual_8/conv2d_19/Conv2D/ReadVariableOp,^residual_8/conv2d_20/BiasAdd/ReadVariableOp+^residual_8/conv2d_20/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2^
-instance_normalization/Reshape/ReadVariableOp-instance_normalization/Reshape/ReadVariableOp2b
/instance_normalization/Reshape_1/ReadVariableOp/instance_normalization/Reshape_1/ReadVariableOp2b
/instance_normalization_1/Reshape/ReadVariableOp/instance_normalization_1/Reshape/ReadVariableOp2f
1instance_normalization_1/Reshape_1/ReadVariableOp1instance_normalization_1/Reshape_1/ReadVariableOp2b
/instance_normalization_2/Reshape/ReadVariableOp/instance_normalization_2/Reshape/ReadVariableOp2f
1instance_normalization_2/Reshape_1/ReadVariableOp1instance_normalization_2/Reshape_1/ReadVariableOp2b
/instance_normalization_3/Reshape/ReadVariableOp/instance_normalization_3/Reshape/ReadVariableOp2f
1instance_normalization_3/Reshape_1/ReadVariableOp1instance_normalization_3/Reshape_1/ReadVariableOp2b
/instance_normalization_4/Reshape/ReadVariableOp/instance_normalization_4/Reshape/ReadVariableOp2f
1instance_normalization_4/Reshape_1/ReadVariableOp1instance_normalization_4/Reshape_1/ReadVariableOp2T
(residual/conv2d_3/BiasAdd/ReadVariableOp(residual/conv2d_3/BiasAdd/ReadVariableOp2R
'residual/conv2d_3/Conv2D/ReadVariableOp'residual/conv2d_3/Conv2D/ReadVariableOp2T
(residual/conv2d_4/BiasAdd/ReadVariableOp(residual/conv2d_4/BiasAdd/ReadVariableOp2R
'residual/conv2d_4/Conv2D/ReadVariableOp'residual/conv2d_4/Conv2D/ReadVariableOp2X
*residual_1/conv2d_5/BiasAdd/ReadVariableOp*residual_1/conv2d_5/BiasAdd/ReadVariableOp2V
)residual_1/conv2d_5/Conv2D/ReadVariableOp)residual_1/conv2d_5/Conv2D/ReadVariableOp2X
*residual_1/conv2d_6/BiasAdd/ReadVariableOp*residual_1/conv2d_6/BiasAdd/ReadVariableOp2V
)residual_1/conv2d_6/Conv2D/ReadVariableOp)residual_1/conv2d_6/Conv2D/ReadVariableOp2X
*residual_2/conv2d_7/BiasAdd/ReadVariableOp*residual_2/conv2d_7/BiasAdd/ReadVariableOp2V
)residual_2/conv2d_7/Conv2D/ReadVariableOp)residual_2/conv2d_7/Conv2D/ReadVariableOp2X
*residual_2/conv2d_8/BiasAdd/ReadVariableOp*residual_2/conv2d_8/BiasAdd/ReadVariableOp2V
)residual_2/conv2d_8/Conv2D/ReadVariableOp)residual_2/conv2d_8/Conv2D/ReadVariableOp2Z
+residual_3/conv2d_10/BiasAdd/ReadVariableOp+residual_3/conv2d_10/BiasAdd/ReadVariableOp2X
*residual_3/conv2d_10/Conv2D/ReadVariableOp*residual_3/conv2d_10/Conv2D/ReadVariableOp2X
*residual_3/conv2d_9/BiasAdd/ReadVariableOp*residual_3/conv2d_9/BiasAdd/ReadVariableOp2V
)residual_3/conv2d_9/Conv2D/ReadVariableOp)residual_3/conv2d_9/Conv2D/ReadVariableOp2Z
+residual_4/conv2d_11/BiasAdd/ReadVariableOp+residual_4/conv2d_11/BiasAdd/ReadVariableOp2X
*residual_4/conv2d_11/Conv2D/ReadVariableOp*residual_4/conv2d_11/Conv2D/ReadVariableOp2Z
+residual_4/conv2d_12/BiasAdd/ReadVariableOp+residual_4/conv2d_12/BiasAdd/ReadVariableOp2X
*residual_4/conv2d_12/Conv2D/ReadVariableOp*residual_4/conv2d_12/Conv2D/ReadVariableOp2Z
+residual_5/conv2d_13/BiasAdd/ReadVariableOp+residual_5/conv2d_13/BiasAdd/ReadVariableOp2X
*residual_5/conv2d_13/Conv2D/ReadVariableOp*residual_5/conv2d_13/Conv2D/ReadVariableOp2Z
+residual_5/conv2d_14/BiasAdd/ReadVariableOp+residual_5/conv2d_14/BiasAdd/ReadVariableOp2X
*residual_5/conv2d_14/Conv2D/ReadVariableOp*residual_5/conv2d_14/Conv2D/ReadVariableOp2Z
+residual_6/conv2d_15/BiasAdd/ReadVariableOp+residual_6/conv2d_15/BiasAdd/ReadVariableOp2X
*residual_6/conv2d_15/Conv2D/ReadVariableOp*residual_6/conv2d_15/Conv2D/ReadVariableOp2Z
+residual_6/conv2d_16/BiasAdd/ReadVariableOp+residual_6/conv2d_16/BiasAdd/ReadVariableOp2X
*residual_6/conv2d_16/Conv2D/ReadVariableOp*residual_6/conv2d_16/Conv2D/ReadVariableOp2Z
+residual_7/conv2d_17/BiasAdd/ReadVariableOp+residual_7/conv2d_17/BiasAdd/ReadVariableOp2X
*residual_7/conv2d_17/Conv2D/ReadVariableOp*residual_7/conv2d_17/Conv2D/ReadVariableOp2Z
+residual_7/conv2d_18/BiasAdd/ReadVariableOp+residual_7/conv2d_18/BiasAdd/ReadVariableOp2X
*residual_7/conv2d_18/Conv2D/ReadVariableOp*residual_7/conv2d_18/Conv2D/ReadVariableOp2Z
+residual_8/conv2d_19/BiasAdd/ReadVariableOp+residual_8/conv2d_19/BiasAdd/ReadVariableOp2X
*residual_8/conv2d_19/Conv2D/ReadVariableOp*residual_8/conv2d_19/Conv2D/ReadVariableOp2Z
+residual_8/conv2d_20/BiasAdd/ReadVariableOp+residual_8/conv2d_20/BiasAdd/ReadVariableOp2X
*residual_8/conv2d_20/Conv2D/ReadVariableOp*residual_8/conv2d_20/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
F__inference_residual_8_layer_call_and_return_conditional_losses_744612

inputsD
(conv2d_19_conv2d_readvariableop_resource:��8
)conv2d_19_biasadd_readvariableop_resource:	�D
(conv2d_20_conv2d_readvariableop_resource:��8
)conv2d_20_biasadd_readvariableop_resource:	�
identity�� conv2d_19/BiasAdd/ReadVariableOp�conv2d_19/Conv2D/ReadVariableOp� conv2d_20/BiasAdd/ReadVariableOp�conv2d_20/Conv2D/ReadVariableOp�
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_19/Conv2DConv2Dinputs'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�m
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_20/Conv2DConv2Dconv2d_19/Relu:activations:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�o
add/addAddV2conv2d_20/BiasAdd:output:0inputs*
T0*0
_output_shapes
:���������@@�Z

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:���������@@�p
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@��
NoOpNoOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
��
�(
"__inference__traced_restore_745746
file_prefix8
assignvariableop_conv2d_kernel:@,
assignvariableop_1_conv2d_bias:@=
/assignvariableop_2_instance_normalization_gamma:@<
.assignvariableop_3_instance_normalization_beta:@=
"assignvariableop_4_conv2d_1_kernel:@�/
 assignvariableop_5_conv2d_1_bias:	�@
1assignvariableop_6_instance_normalization_1_gamma:	�?
0assignvariableop_7_instance_normalization_1_beta:	�>
"assignvariableop_8_conv2d_2_kernel:��/
 assignvariableop_9_conv2d_2_bias:	�A
2assignvariableop_10_instance_normalization_2_gamma:	�@
1assignvariableop_11_instance_normalization_2_beta:	�G
+assignvariableop_12_conv2d_transpose_kernel:��8
)assignvariableop_13_conv2d_transpose_bias:	�A
2assignvariableop_14_instance_normalization_3_gamma:	�@
1assignvariableop_15_instance_normalization_3_beta:	�H
-assignvariableop_16_conv2d_transpose_1_kernel:@�9
+assignvariableop_17_conv2d_transpose_1_bias:@@
2assignvariableop_18_instance_normalization_4_gamma:@?
1assignvariableop_19_instance_normalization_4_beta:@>
$assignvariableop_20_conv2d_21_kernel:@0
"assignvariableop_21_conv2d_21_bias:H
,assignvariableop_22_residual_conv2d_3_kernel:��9
*assignvariableop_23_residual_conv2d_3_bias:	�H
,assignvariableop_24_residual_conv2d_4_kernel:��9
*assignvariableop_25_residual_conv2d_4_bias:	�J
.assignvariableop_26_residual_1_conv2d_5_kernel:��;
,assignvariableop_27_residual_1_conv2d_5_bias:	�J
.assignvariableop_28_residual_1_conv2d_6_kernel:��;
,assignvariableop_29_residual_1_conv2d_6_bias:	�J
.assignvariableop_30_residual_2_conv2d_7_kernel:��;
,assignvariableop_31_residual_2_conv2d_7_bias:	�J
.assignvariableop_32_residual_2_conv2d_8_kernel:��;
,assignvariableop_33_residual_2_conv2d_8_bias:	�J
.assignvariableop_34_residual_3_conv2d_9_kernel:��;
,assignvariableop_35_residual_3_conv2d_9_bias:	�K
/assignvariableop_36_residual_3_conv2d_10_kernel:��<
-assignvariableop_37_residual_3_conv2d_10_bias:	�K
/assignvariableop_38_residual_4_conv2d_11_kernel:��<
-assignvariableop_39_residual_4_conv2d_11_bias:	�K
/assignvariableop_40_residual_4_conv2d_12_kernel:��<
-assignvariableop_41_residual_4_conv2d_12_bias:	�K
/assignvariableop_42_residual_5_conv2d_13_kernel:��<
-assignvariableop_43_residual_5_conv2d_13_bias:	�K
/assignvariableop_44_residual_5_conv2d_14_kernel:��<
-assignvariableop_45_residual_5_conv2d_14_bias:	�K
/assignvariableop_46_residual_6_conv2d_15_kernel:��<
-assignvariableop_47_residual_6_conv2d_15_bias:	�K
/assignvariableop_48_residual_6_conv2d_16_kernel:��<
-assignvariableop_49_residual_6_conv2d_16_bias:	�K
/assignvariableop_50_residual_7_conv2d_17_kernel:��<
-assignvariableop_51_residual_7_conv2d_17_bias:	�K
/assignvariableop_52_residual_7_conv2d_18_kernel:��<
-assignvariableop_53_residual_7_conv2d_18_bias:	�K
/assignvariableop_54_residual_8_conv2d_19_kernel:��<
-assignvariableop_55_residual_8_conv2d_19_bias:	�K
/assignvariableop_56_residual_8_conv2d_20_kernel:��<
-assignvariableop_57_residual_8_conv2d_20_bias:	�
identity_59��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*�
value�B�;B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*�
value�B~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*I
dtypes?
=2;[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp/assignvariableop_2_instance_normalization_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp.assignvariableop_3_instance_normalization_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_1_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_1_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp1assignvariableop_6_instance_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp0assignvariableop_7_instance_normalization_1_betaIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_2_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_2_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp2assignvariableop_10_instance_normalization_2_gammaIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp1assignvariableop_11_instance_normalization_2_betaIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp+assignvariableop_12_conv2d_transpose_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp)assignvariableop_13_conv2d_transpose_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp2assignvariableop_14_instance_normalization_3_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp1assignvariableop_15_instance_normalization_3_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp-assignvariableop_16_conv2d_transpose_1_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp+assignvariableop_17_conv2d_transpose_1_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp2assignvariableop_18_instance_normalization_4_gammaIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp1assignvariableop_19_instance_normalization_4_betaIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp$assignvariableop_20_conv2d_21_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp"assignvariableop_21_conv2d_21_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp,assignvariableop_22_residual_conv2d_3_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_residual_conv2d_3_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp,assignvariableop_24_residual_conv2d_4_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp*assignvariableop_25_residual_conv2d_4_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp.assignvariableop_26_residual_1_conv2d_5_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp,assignvariableop_27_residual_1_conv2d_5_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp.assignvariableop_28_residual_1_conv2d_6_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp,assignvariableop_29_residual_1_conv2d_6_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp.assignvariableop_30_residual_2_conv2d_7_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp,assignvariableop_31_residual_2_conv2d_7_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp.assignvariableop_32_residual_2_conv2d_8_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp,assignvariableop_33_residual_2_conv2d_8_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp.assignvariableop_34_residual_3_conv2d_9_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp,assignvariableop_35_residual_3_conv2d_9_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp/assignvariableop_36_residual_3_conv2d_10_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp-assignvariableop_37_residual_3_conv2d_10_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp/assignvariableop_38_residual_4_conv2d_11_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp-assignvariableop_39_residual_4_conv2d_11_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp/assignvariableop_40_residual_4_conv2d_12_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp-assignvariableop_41_residual_4_conv2d_12_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp/assignvariableop_42_residual_5_conv2d_13_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp-assignvariableop_43_residual_5_conv2d_13_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp/assignvariableop_44_residual_5_conv2d_14_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp-assignvariableop_45_residual_5_conv2d_14_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp/assignvariableop_46_residual_6_conv2d_15_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp-assignvariableop_47_residual_6_conv2d_15_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp/assignvariableop_48_residual_6_conv2d_16_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp-assignvariableop_49_residual_6_conv2d_16_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp/assignvariableop_50_residual_7_conv2d_17_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp-assignvariableop_51_residual_7_conv2d_17_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp/assignvariableop_52_residual_7_conv2d_18_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp-assignvariableop_53_residual_7_conv2d_18_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp/assignvariableop_54_residual_8_conv2d_19_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp-assignvariableop_55_residual_8_conv2d_19_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp/assignvariableop_56_residual_8_conv2d_20_kernelIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp-assignvariableop_57_residual_8_conv2d_20_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_58Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_59IdentityIdentity_58:output:0^NoOp_1*
T0*
_output_shapes
: �

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_59Identity_59:output:0*�
_input_shapesx
v: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
��
�
F__inference_sequential_layer_call_and_return_conditional_losses_742431

inputs'
conv2d_742289:@
conv2d_742291:@+
instance_normalization_742294:@+
instance_normalization_742296:@*
conv2d_1_742300:@�
conv2d_1_742302:	�.
instance_normalization_1_742305:	�.
instance_normalization_1_742307:	�+
conv2d_2_742311:��
conv2d_2_742313:	�.
instance_normalization_2_742316:	�.
instance_normalization_2_742318:	�+
residual_742322:��
residual_742324:	�+
residual_742326:��
residual_742328:	�-
residual_1_742331:�� 
residual_1_742333:	�-
residual_1_742335:�� 
residual_1_742337:	�-
residual_2_742340:�� 
residual_2_742342:	�-
residual_2_742344:�� 
residual_2_742346:	�-
residual_3_742349:�� 
residual_3_742351:	�-
residual_3_742353:�� 
residual_3_742355:	�-
residual_4_742358:�� 
residual_4_742360:	�-
residual_4_742362:�� 
residual_4_742364:	�-
residual_5_742367:�� 
residual_5_742369:	�-
residual_5_742371:�� 
residual_5_742373:	�-
residual_6_742376:�� 
residual_6_742378:	�-
residual_6_742380:�� 
residual_6_742382:	�-
residual_7_742385:�� 
residual_7_742387:	�-
residual_7_742389:�� 
residual_7_742391:	�-
residual_8_742394:�� 
residual_8_742396:	�-
residual_8_742398:�� 
residual_8_742400:	�3
conv2d_transpose_742403:��&
conv2d_transpose_742405:	�.
instance_normalization_3_742408:	�.
instance_normalization_3_742410:	�4
conv2d_transpose_1_742414:@�'
conv2d_transpose_1_742416:@-
instance_normalization_4_742419:@-
instance_normalization_4_742421:@*
conv2d_21_742425:@
conv2d_21_742427:
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�!conv2d_21/StatefulPartitionedCall�(conv2d_transpose/StatefulPartitionedCall�*conv2d_transpose_1/StatefulPartitionedCall�.instance_normalization/StatefulPartitionedCall�0instance_normalization_1/StatefulPartitionedCall�0instance_normalization_2/StatefulPartitionedCall�0instance_normalization_3/StatefulPartitionedCall�0instance_normalization_4/StatefulPartitionedCall� residual/StatefulPartitionedCall�"residual_1/StatefulPartitionedCall�"residual_2/StatefulPartitionedCall�"residual_3/StatefulPartitionedCall�"residual_4/StatefulPartitionedCall�"residual_5/StatefulPartitionedCall�"residual_6/StatefulPartitionedCall�"residual_7/StatefulPartitionedCall�"residual_8/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_742289conv2d_742291*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_741445�
.instance_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0instance_normalization_742294instance_normalization_742296*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *[
fVRT
R__inference_instance_normalization_layer_call_and_return_conditional_losses_741494�
re_lu/PartitionedCallPartitionedCall7instance_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_741505�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_742300conv2d_1_742302*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_741517�
0instance_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0instance_normalization_1_742305instance_normalization_1_742307*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *]
fXRV
T__inference_instance_normalization_1_layer_call_and_return_conditional_losses_741566�
re_lu_1/PartitionedCallPartitionedCall9instance_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *L
fGRE
C__inference_re_lu_1_layer_call_and_return_conditional_losses_741577�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_2_742311conv2d_2_742313*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_741589�
0instance_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0instance_normalization_2_742316instance_normalization_2_742318*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *]
fXRV
T__inference_instance_normalization_2_layer_call_and_return_conditional_losses_741638�
re_lu_2/PartitionedCallPartitionedCall9instance_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_741649�
 residual/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0residual_742322residual_742324residual_742326residual_742328*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_residual_layer_call_and_return_conditional_losses_740717�
"residual_1/StatefulPartitionedCallStatefulPartitionedCall)residual/StatefulPartitionedCall:output:0residual_1_742331residual_1_742333residual_1_742335residual_1_742337*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_1_layer_call_and_return_conditional_losses_740791�
"residual_2/StatefulPartitionedCallStatefulPartitionedCall+residual_1/StatefulPartitionedCall:output:0residual_2_742340residual_2_742342residual_2_742344residual_2_742346*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_2_layer_call_and_return_conditional_losses_740865�
"residual_3/StatefulPartitionedCallStatefulPartitionedCall+residual_2/StatefulPartitionedCall:output:0residual_3_742349residual_3_742351residual_3_742353residual_3_742355*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_3_layer_call_and_return_conditional_losses_740939�
"residual_4/StatefulPartitionedCallStatefulPartitionedCall+residual_3/StatefulPartitionedCall:output:0residual_4_742358residual_4_742360residual_4_742362residual_4_742364*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_4_layer_call_and_return_conditional_losses_741013�
"residual_5/StatefulPartitionedCallStatefulPartitionedCall+residual_4/StatefulPartitionedCall:output:0residual_5_742367residual_5_742369residual_5_742371residual_5_742373*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_5_layer_call_and_return_conditional_losses_741087�
"residual_6/StatefulPartitionedCallStatefulPartitionedCall+residual_5/StatefulPartitionedCall:output:0residual_6_742376residual_6_742378residual_6_742380residual_6_742382*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_6_layer_call_and_return_conditional_losses_741161�
"residual_7/StatefulPartitionedCallStatefulPartitionedCall+residual_6/StatefulPartitionedCall:output:0residual_7_742385residual_7_742387residual_7_742389residual_7_742391*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_7_layer_call_and_return_conditional_losses_741235�
"residual_8/StatefulPartitionedCallStatefulPartitionedCall+residual_7/StatefulPartitionedCall:output:0residual_8_742394residual_8_742396residual_8_742398residual_8_742400*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *O
fJRH
F__inference_residual_8_layer_call_and_return_conditional_losses_741309�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall+residual_8/StatefulPartitionedCall:output:0conv2d_transpose_742403conv2d_transpose_742405*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_741377�
0instance_normalization_3/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0instance_normalization_3_742408instance_normalization_3_742410*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *]
fXRV
T__inference_instance_normalization_3_layer_call_and_return_conditional_losses_741780�
re_lu_3/PartitionedCallPartitionedCall9instance_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_741791�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv2d_transpose_1_742414conv2d_transpose_1_742416*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_741421�
0instance_normalization_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0instance_normalization_4_742419instance_normalization_4_742421*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *]
fXRV
T__inference_instance_normalization_4_layer_call_and_return_conditional_losses_741841�
re_lu_4/PartitionedCallPartitionedCall9instance_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2 *0,1,2J 8� *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_741852�
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv2d_21_742425conv2d_21_742427*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_741865�
IdentityIdentity*conv2d_21/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall/^instance_normalization/StatefulPartitionedCall1^instance_normalization_1/StatefulPartitionedCall1^instance_normalization_2/StatefulPartitionedCall1^instance_normalization_3/StatefulPartitionedCall1^instance_normalization_4/StatefulPartitionedCall!^residual/StatefulPartitionedCall#^residual_1/StatefulPartitionedCall#^residual_2/StatefulPartitionedCall#^residual_3/StatefulPartitionedCall#^residual_4/StatefulPartitionedCall#^residual_5/StatefulPartitionedCall#^residual_6/StatefulPartitionedCall#^residual_7/StatefulPartitionedCall#^residual_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2`
.instance_normalization/StatefulPartitionedCall.instance_normalization/StatefulPartitionedCall2d
0instance_normalization_1/StatefulPartitionedCall0instance_normalization_1/StatefulPartitionedCall2d
0instance_normalization_2/StatefulPartitionedCall0instance_normalization_2/StatefulPartitionedCall2d
0instance_normalization_3/StatefulPartitionedCall0instance_normalization_3/StatefulPartitionedCall2d
0instance_normalization_4/StatefulPartitionedCall0instance_normalization_4/StatefulPartitionedCall2D
 residual/StatefulPartitionedCall residual/StatefulPartitionedCall2H
"residual_1/StatefulPartitionedCall"residual_1/StatefulPartitionedCall2H
"residual_2/StatefulPartitionedCall"residual_2/StatefulPartitionedCall2H
"residual_3/StatefulPartitionedCall"residual_3/StatefulPartitionedCall2H
"residual_4/StatefulPartitionedCall"residual_4/StatefulPartitionedCall2H
"residual_5/StatefulPartitionedCall"residual_5/StatefulPartitionedCall2H
"residual_6/StatefulPartitionedCall"residual_6/StatefulPartitionedCall2H
"residual_7/StatefulPartitionedCall"residual_7/StatefulPartitionedCall2H
"residual_8/StatefulPartitionedCall"residual_8/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
*__inference_conv2d_18_layer_call_fn_745142

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_741226x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������@@�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_743035
input_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�&

unknown_25:��

unknown_26:	�&

unknown_27:��

unknown_28:	�&

unknown_29:��

unknown_30:	�&

unknown_31:��

unknown_32:	�&

unknown_33:��

unknown_34:	�&

unknown_35:��

unknown_36:	�&

unknown_37:��

unknown_38:	�&

unknown_39:��

unknown_40:	�&

unknown_41:��

unknown_42:	�&

unknown_43:��

unknown_44:	�&

unknown_45:��

unknown_46:	�&

unknown_47:��

unknown_48:	�

unknown_49:	�

unknown_50:	�%

unknown_51:@�

unknown_52:@

unknown_53:@

unknown_54:@$

unknown_55:@

unknown_56:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:*6
config_proto&$

CPU

GPU2 *0,1,2J 8� **
f%R#
!__inference__wrapped_model_740677y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
_
C__inference_re_lu_1_layer_call_and_return_conditional_losses_744243

inputs
identityQ
ReluReluinputs*
T0*2
_output_shapes 
:������������e
IdentityIdentityRelu:activations:0*
T0*2
_output_shapes 
:������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������:Z V
2
_output_shapes 
:������������
 
_user_specified_nameinputs
�
�
F__inference_residual_2_layer_call_and_return_conditional_losses_740865

inputs+
conv2d_7_740841:��
conv2d_7_740843:	�+
conv2d_8_740857:��
conv2d_8_740859:	�
identity�� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_7_740841conv2d_7_740843*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_740840�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0conv2d_8_740857conv2d_8_740859*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_740856~
add/addAddV2)conv2d_8/StatefulPartitionedCall:output:0inputs*
T0*0
_output_shapes
:���������@@�Z

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:���������@@�p
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@��
NoOpNoOp!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
��
�?
!__inference__wrapped_model_740677
input_1J
0sequential_conv2d_conv2d_readvariableop_resource:@?
1sequential_conv2d_biasadd_readvariableop_resource:@O
Asequential_instance_normalization_reshape_readvariableop_resource:@Q
Csequential_instance_normalization_reshape_1_readvariableop_resource:@M
2sequential_conv2d_1_conv2d_readvariableop_resource:@�B
3sequential_conv2d_1_biasadd_readvariableop_resource:	�R
Csequential_instance_normalization_1_reshape_readvariableop_resource:	�T
Esequential_instance_normalization_1_reshape_1_readvariableop_resource:	�N
2sequential_conv2d_2_conv2d_readvariableop_resource:��B
3sequential_conv2d_2_biasadd_readvariableop_resource:	�R
Csequential_instance_normalization_2_reshape_readvariableop_resource:	�T
Esequential_instance_normalization_2_reshape_1_readvariableop_resource:	�W
;sequential_residual_conv2d_3_conv2d_readvariableop_resource:��K
<sequential_residual_conv2d_3_biasadd_readvariableop_resource:	�W
;sequential_residual_conv2d_4_conv2d_readvariableop_resource:��K
<sequential_residual_conv2d_4_biasadd_readvariableop_resource:	�Y
=sequential_residual_1_conv2d_5_conv2d_readvariableop_resource:��M
>sequential_residual_1_conv2d_5_biasadd_readvariableop_resource:	�Y
=sequential_residual_1_conv2d_6_conv2d_readvariableop_resource:��M
>sequential_residual_1_conv2d_6_biasadd_readvariableop_resource:	�Y
=sequential_residual_2_conv2d_7_conv2d_readvariableop_resource:��M
>sequential_residual_2_conv2d_7_biasadd_readvariableop_resource:	�Y
=sequential_residual_2_conv2d_8_conv2d_readvariableop_resource:��M
>sequential_residual_2_conv2d_8_biasadd_readvariableop_resource:	�Y
=sequential_residual_3_conv2d_9_conv2d_readvariableop_resource:��M
>sequential_residual_3_conv2d_9_biasadd_readvariableop_resource:	�Z
>sequential_residual_3_conv2d_10_conv2d_readvariableop_resource:��N
?sequential_residual_3_conv2d_10_biasadd_readvariableop_resource:	�Z
>sequential_residual_4_conv2d_11_conv2d_readvariableop_resource:��N
?sequential_residual_4_conv2d_11_biasadd_readvariableop_resource:	�Z
>sequential_residual_4_conv2d_12_conv2d_readvariableop_resource:��N
?sequential_residual_4_conv2d_12_biasadd_readvariableop_resource:	�Z
>sequential_residual_5_conv2d_13_conv2d_readvariableop_resource:��N
?sequential_residual_5_conv2d_13_biasadd_readvariableop_resource:	�Z
>sequential_residual_5_conv2d_14_conv2d_readvariableop_resource:��N
?sequential_residual_5_conv2d_14_biasadd_readvariableop_resource:	�Z
>sequential_residual_6_conv2d_15_conv2d_readvariableop_resource:��N
?sequential_residual_6_conv2d_15_biasadd_readvariableop_resource:	�Z
>sequential_residual_6_conv2d_16_conv2d_readvariableop_resource:��N
?sequential_residual_6_conv2d_16_biasadd_readvariableop_resource:	�Z
>sequential_residual_7_conv2d_17_conv2d_readvariableop_resource:��N
?sequential_residual_7_conv2d_17_biasadd_readvariableop_resource:	�Z
>sequential_residual_7_conv2d_18_conv2d_readvariableop_resource:��N
?sequential_residual_7_conv2d_18_biasadd_readvariableop_resource:	�Z
>sequential_residual_8_conv2d_19_conv2d_readvariableop_resource:��N
?sequential_residual_8_conv2d_19_biasadd_readvariableop_resource:	�Z
>sequential_residual_8_conv2d_20_conv2d_readvariableop_resource:��N
?sequential_residual_8_conv2d_20_biasadd_readvariableop_resource:	�`
Dsequential_conv2d_transpose_conv2d_transpose_readvariableop_resource:��J
;sequential_conv2d_transpose_biasadd_readvariableop_resource:	�R
Csequential_instance_normalization_3_reshape_readvariableop_resource:	�T
Esequential_instance_normalization_3_reshape_1_readvariableop_resource:	�a
Fsequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@�K
=sequential_conv2d_transpose_1_biasadd_readvariableop_resource:@Q
Csequential_instance_normalization_4_reshape_readvariableop_resource:@S
Esequential_instance_normalization_4_reshape_1_readvariableop_resource:@M
3sequential_conv2d_21_conv2d_readvariableop_resource:@B
4sequential_conv2d_21_biasadd_readvariableop_resource:
identity��(sequential/conv2d/BiasAdd/ReadVariableOp�'sequential/conv2d/Conv2D/ReadVariableOp�*sequential/conv2d_1/BiasAdd/ReadVariableOp�)sequential/conv2d_1/Conv2D/ReadVariableOp�*sequential/conv2d_2/BiasAdd/ReadVariableOp�)sequential/conv2d_2/Conv2D/ReadVariableOp�+sequential/conv2d_21/BiasAdd/ReadVariableOp�*sequential/conv2d_21/Conv2D/ReadVariableOp�2sequential/conv2d_transpose/BiasAdd/ReadVariableOp�;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp�4sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp�=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp�8sequential/instance_normalization/Reshape/ReadVariableOp�:sequential/instance_normalization/Reshape_1/ReadVariableOp�:sequential/instance_normalization_1/Reshape/ReadVariableOp�<sequential/instance_normalization_1/Reshape_1/ReadVariableOp�:sequential/instance_normalization_2/Reshape/ReadVariableOp�<sequential/instance_normalization_2/Reshape_1/ReadVariableOp�:sequential/instance_normalization_3/Reshape/ReadVariableOp�<sequential/instance_normalization_3/Reshape_1/ReadVariableOp�:sequential/instance_normalization_4/Reshape/ReadVariableOp�<sequential/instance_normalization_4/Reshape_1/ReadVariableOp�3sequential/residual/conv2d_3/BiasAdd/ReadVariableOp�2sequential/residual/conv2d_3/Conv2D/ReadVariableOp�3sequential/residual/conv2d_4/BiasAdd/ReadVariableOp�2sequential/residual/conv2d_4/Conv2D/ReadVariableOp�5sequential/residual_1/conv2d_5/BiasAdd/ReadVariableOp�4sequential/residual_1/conv2d_5/Conv2D/ReadVariableOp�5sequential/residual_1/conv2d_6/BiasAdd/ReadVariableOp�4sequential/residual_1/conv2d_6/Conv2D/ReadVariableOp�5sequential/residual_2/conv2d_7/BiasAdd/ReadVariableOp�4sequential/residual_2/conv2d_7/Conv2D/ReadVariableOp�5sequential/residual_2/conv2d_8/BiasAdd/ReadVariableOp�4sequential/residual_2/conv2d_8/Conv2D/ReadVariableOp�6sequential/residual_3/conv2d_10/BiasAdd/ReadVariableOp�5sequential/residual_3/conv2d_10/Conv2D/ReadVariableOp�5sequential/residual_3/conv2d_9/BiasAdd/ReadVariableOp�4sequential/residual_3/conv2d_9/Conv2D/ReadVariableOp�6sequential/residual_4/conv2d_11/BiasAdd/ReadVariableOp�5sequential/residual_4/conv2d_11/Conv2D/ReadVariableOp�6sequential/residual_4/conv2d_12/BiasAdd/ReadVariableOp�5sequential/residual_4/conv2d_12/Conv2D/ReadVariableOp�6sequential/residual_5/conv2d_13/BiasAdd/ReadVariableOp�5sequential/residual_5/conv2d_13/Conv2D/ReadVariableOp�6sequential/residual_5/conv2d_14/BiasAdd/ReadVariableOp�5sequential/residual_5/conv2d_14/Conv2D/ReadVariableOp�6sequential/residual_6/conv2d_15/BiasAdd/ReadVariableOp�5sequential/residual_6/conv2d_15/Conv2D/ReadVariableOp�6sequential/residual_6/conv2d_16/BiasAdd/ReadVariableOp�5sequential/residual_6/conv2d_16/Conv2D/ReadVariableOp�6sequential/residual_7/conv2d_17/BiasAdd/ReadVariableOp�5sequential/residual_7/conv2d_17/Conv2D/ReadVariableOp�6sequential/residual_7/conv2d_18/BiasAdd/ReadVariableOp�5sequential/residual_7/conv2d_18/Conv2D/ReadVariableOp�6sequential/residual_8/conv2d_19/BiasAdd/ReadVariableOp�5sequential/residual_8/conv2d_19/Conv2D/ReadVariableOp�6sequential/residual_8/conv2d_20/BiasAdd/ReadVariableOp�5sequential/residual_8/conv2d_20/Conv2D/ReadVariableOp�
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
sequential/conv2d/Conv2DConv2Dinput_1/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@�
'sequential/instance_normalization/ShapeShape"sequential/conv2d/BiasAdd:output:0*
T0*
_output_shapes
::��
5sequential/instance_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
7sequential/instance_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
7sequential/instance_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
/sequential/instance_normalization/strided_sliceStridedSlice0sequential/instance_normalization/Shape:output:0>sequential/instance_normalization/strided_slice/stack:output:0@sequential/instance_normalization/strided_slice/stack_1:output:0@sequential/instance_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
7sequential/instance_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
9sequential/instance_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9sequential/instance_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1sequential/instance_normalization/strided_slice_1StridedSlice0sequential/instance_normalization/Shape:output:0@sequential/instance_normalization/strided_slice_1/stack:output:0Bsequential/instance_normalization/strided_slice_1/stack_1:output:0Bsequential/instance_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
7sequential/instance_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
9sequential/instance_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9sequential/instance_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1sequential/instance_normalization/strided_slice_2StridedSlice0sequential/instance_normalization/Shape:output:0@sequential/instance_normalization/strided_slice_2/stack:output:0Bsequential/instance_normalization/strided_slice_2/stack_1:output:0Bsequential/instance_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
7sequential/instance_normalization/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:�
9sequential/instance_normalization/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9sequential/instance_normalization/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1sequential/instance_normalization/strided_slice_3StridedSlice0sequential/instance_normalization/Shape:output:0@sequential/instance_normalization/strided_slice_3/stack:output:0Bsequential/instance_normalization/strided_slice_3/stack_1:output:0Bsequential/instance_normalization/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
@sequential/instance_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
.sequential/instance_normalization/moments/meanMean"sequential/conv2d/BiasAdd:output:0Isequential/instance_normalization/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������@*
	keep_dims(�
6sequential/instance_normalization/moments/StopGradientStopGradient7sequential/instance_normalization/moments/mean:output:0*
T0*/
_output_shapes
:���������@�
;sequential/instance_normalization/moments/SquaredDifferenceSquaredDifference"sequential/conv2d/BiasAdd:output:0?sequential/instance_normalization/moments/StopGradient:output:0*
T0*1
_output_shapes
:�����������@�
Dsequential/instance_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
2sequential/instance_normalization/moments/varianceMean?sequential/instance_normalization/moments/SquaredDifference:z:0Msequential/instance_normalization/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:���������@*
	keep_dims(�
8sequential/instance_normalization/Reshape/ReadVariableOpReadVariableOpAsequential_instance_normalization_reshape_readvariableop_resource*
_output_shapes
:@*
dtype0�
/sequential/instance_normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
)sequential/instance_normalization/ReshapeReshape@sequential/instance_normalization/Reshape/ReadVariableOp:value:08sequential/instance_normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:@�
:sequential/instance_normalization/Reshape_1/ReadVariableOpReadVariableOpCsequential_instance_normalization_reshape_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
1sequential/instance_normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
+sequential/instance_normalization/Reshape_1ReshapeBsequential/instance_normalization/Reshape_1/ReadVariableOp:value:0:sequential/instance_normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:@v
1sequential/instance_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
/sequential/instance_normalization/batchnorm/addAddV2;sequential/instance_normalization/moments/variance:output:0:sequential/instance_normalization/batchnorm/add/y:output:0*
T0*/
_output_shapes
:���������@�
1sequential/instance_normalization/batchnorm/RsqrtRsqrt3sequential/instance_normalization/batchnorm/add:z:0*
T0*/
_output_shapes
:���������@�
/sequential/instance_normalization/batchnorm/mulMul5sequential/instance_normalization/batchnorm/Rsqrt:y:02sequential/instance_normalization/Reshape:output:0*
T0*/
_output_shapes
:���������@�
1sequential/instance_normalization/batchnorm/mul_1Mul"sequential/conv2d/BiasAdd:output:03sequential/instance_normalization/batchnorm/mul:z:0*
T0*1
_output_shapes
:�����������@�
1sequential/instance_normalization/batchnorm/mul_2Mul7sequential/instance_normalization/moments/mean:output:03sequential/instance_normalization/batchnorm/mul:z:0*
T0*/
_output_shapes
:���������@�
/sequential/instance_normalization/batchnorm/subSub4sequential/instance_normalization/Reshape_1:output:05sequential/instance_normalization/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:���������@�
1sequential/instance_normalization/batchnorm/add_1AddV25sequential/instance_normalization/batchnorm/mul_1:z:03sequential/instance_normalization/batchnorm/sub:z:0*
T0*1
_output_shapes
:�����������@�
sequential/re_lu/ReluRelu5sequential/instance_normalization/batchnorm/add_1:z:0*
T0*1
_output_shapes
:�����������@�
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������*
paddingSAME*
strides
�
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:�������������
)sequential/instance_normalization_1/ShapeShape$sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
::���
7sequential/instance_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
9sequential/instance_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9sequential/instance_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1sequential/instance_normalization_1/strided_sliceStridedSlice2sequential/instance_normalization_1/Shape:output:0@sequential/instance_normalization_1/strided_slice/stack:output:0Bsequential/instance_normalization_1/strided_slice/stack_1:output:0Bsequential/instance_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
9sequential/instance_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3sequential/instance_normalization_1/strided_slice_1StridedSlice2sequential/instance_normalization_1/Shape:output:0Bsequential/instance_normalization_1/strided_slice_1/stack:output:0Dsequential/instance_normalization_1/strided_slice_1/stack_1:output:0Dsequential/instance_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
9sequential/instance_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3sequential/instance_normalization_1/strided_slice_2StridedSlice2sequential/instance_normalization_1/Shape:output:0Bsequential/instance_normalization_1/strided_slice_2/stack:output:0Dsequential/instance_normalization_1/strided_slice_2/stack_1:output:0Dsequential/instance_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
9sequential/instance_normalization_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3sequential/instance_normalization_1/strided_slice_3StridedSlice2sequential/instance_normalization_1/Shape:output:0Bsequential/instance_normalization_1/strided_slice_3/stack:output:0Dsequential/instance_normalization_1/strided_slice_3/stack_1:output:0Dsequential/instance_normalization_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Bsequential/instance_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
0sequential/instance_normalization_1/moments/meanMean$sequential/conv2d_1/BiasAdd:output:0Ksequential/instance_normalization_1/moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
8sequential/instance_normalization_1/moments/StopGradientStopGradient9sequential/instance_normalization_1/moments/mean:output:0*
T0*0
_output_shapes
:�����������
=sequential/instance_normalization_1/moments/SquaredDifferenceSquaredDifference$sequential/conv2d_1/BiasAdd:output:0Asequential/instance_normalization_1/moments/StopGradient:output:0*
T0*2
_output_shapes 
:�������������
Fsequential/instance_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
4sequential/instance_normalization_1/moments/varianceMeanAsequential/instance_normalization_1/moments/SquaredDifference:z:0Osequential/instance_normalization_1/moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
:sequential/instance_normalization_1/Reshape/ReadVariableOpReadVariableOpCsequential_instance_normalization_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
1sequential/instance_normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
+sequential/instance_normalization_1/ReshapeReshapeBsequential/instance_normalization_1/Reshape/ReadVariableOp:value:0:sequential/instance_normalization_1/Reshape/shape:output:0*
T0*'
_output_shapes
:��
<sequential/instance_normalization_1/Reshape_1/ReadVariableOpReadVariableOpEsequential_instance_normalization_1_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
3sequential/instance_normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
-sequential/instance_normalization_1/Reshape_1ReshapeDsequential/instance_normalization_1/Reshape_1/ReadVariableOp:value:0<sequential/instance_normalization_1/Reshape_1/shape:output:0*
T0*'
_output_shapes
:�x
3sequential/instance_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1sequential/instance_normalization_1/batchnorm/addAddV2=sequential/instance_normalization_1/moments/variance:output:0<sequential/instance_normalization_1/batchnorm/add/y:output:0*
T0*0
_output_shapes
:�����������
3sequential/instance_normalization_1/batchnorm/RsqrtRsqrt5sequential/instance_normalization_1/batchnorm/add:z:0*
T0*0
_output_shapes
:�����������
1sequential/instance_normalization_1/batchnorm/mulMul7sequential/instance_normalization_1/batchnorm/Rsqrt:y:04sequential/instance_normalization_1/Reshape:output:0*
T0*0
_output_shapes
:�����������
3sequential/instance_normalization_1/batchnorm/mul_1Mul$sequential/conv2d_1/BiasAdd:output:05sequential/instance_normalization_1/batchnorm/mul:z:0*
T0*2
_output_shapes 
:�������������
3sequential/instance_normalization_1/batchnorm/mul_2Mul9sequential/instance_normalization_1/moments/mean:output:05sequential/instance_normalization_1/batchnorm/mul:z:0*
T0*0
_output_shapes
:�����������
1sequential/instance_normalization_1/batchnorm/subSub6sequential/instance_normalization_1/Reshape_1:output:07sequential/instance_normalization_1/batchnorm/mul_2:z:0*
T0*0
_output_shapes
:�����������
3sequential/instance_normalization_1/batchnorm/add_1AddV27sequential/instance_normalization_1/batchnorm/mul_1:z:05sequential/instance_normalization_1/batchnorm/sub:z:0*
T0*2
_output_shapes 
:�������������
sequential/re_lu_1/ReluRelu7sequential/instance_normalization_1/batchnorm/add_1:z:0*
T0*2
_output_shapes 
:�������������
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
sequential/conv2d_2/Conv2DConv2D%sequential/re_lu_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
)sequential/instance_normalization_2/ShapeShape$sequential/conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
::���
7sequential/instance_normalization_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
9sequential/instance_normalization_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9sequential/instance_normalization_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1sequential/instance_normalization_2/strided_sliceStridedSlice2sequential/instance_normalization_2/Shape:output:0@sequential/instance_normalization_2/strided_slice/stack:output:0Bsequential/instance_normalization_2/strided_slice/stack_1:output:0Bsequential/instance_normalization_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
9sequential/instance_normalization_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3sequential/instance_normalization_2/strided_slice_1StridedSlice2sequential/instance_normalization_2/Shape:output:0Bsequential/instance_normalization_2/strided_slice_1/stack:output:0Dsequential/instance_normalization_2/strided_slice_1/stack_1:output:0Dsequential/instance_normalization_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
9sequential/instance_normalization_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3sequential/instance_normalization_2/strided_slice_2StridedSlice2sequential/instance_normalization_2/Shape:output:0Bsequential/instance_normalization_2/strided_slice_2/stack:output:0Dsequential/instance_normalization_2/strided_slice_2/stack_1:output:0Dsequential/instance_normalization_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
9sequential/instance_normalization_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3sequential/instance_normalization_2/strided_slice_3StridedSlice2sequential/instance_normalization_2/Shape:output:0Bsequential/instance_normalization_2/strided_slice_3/stack:output:0Dsequential/instance_normalization_2/strided_slice_3/stack_1:output:0Dsequential/instance_normalization_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Bsequential/instance_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
0sequential/instance_normalization_2/moments/meanMean$sequential/conv2d_2/BiasAdd:output:0Ksequential/instance_normalization_2/moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
8sequential/instance_normalization_2/moments/StopGradientStopGradient9sequential/instance_normalization_2/moments/mean:output:0*
T0*0
_output_shapes
:�����������
=sequential/instance_normalization_2/moments/SquaredDifferenceSquaredDifference$sequential/conv2d_2/BiasAdd:output:0Asequential/instance_normalization_2/moments/StopGradient:output:0*
T0*0
_output_shapes
:���������@@��
Fsequential/instance_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
4sequential/instance_normalization_2/moments/varianceMeanAsequential/instance_normalization_2/moments/SquaredDifference:z:0Osequential/instance_normalization_2/moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
:sequential/instance_normalization_2/Reshape/ReadVariableOpReadVariableOpCsequential_instance_normalization_2_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
1sequential/instance_normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
+sequential/instance_normalization_2/ReshapeReshapeBsequential/instance_normalization_2/Reshape/ReadVariableOp:value:0:sequential/instance_normalization_2/Reshape/shape:output:0*
T0*'
_output_shapes
:��
<sequential/instance_normalization_2/Reshape_1/ReadVariableOpReadVariableOpEsequential_instance_normalization_2_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
3sequential/instance_normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
-sequential/instance_normalization_2/Reshape_1ReshapeDsequential/instance_normalization_2/Reshape_1/ReadVariableOp:value:0<sequential/instance_normalization_2/Reshape_1/shape:output:0*
T0*'
_output_shapes
:�x
3sequential/instance_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1sequential/instance_normalization_2/batchnorm/addAddV2=sequential/instance_normalization_2/moments/variance:output:0<sequential/instance_normalization_2/batchnorm/add/y:output:0*
T0*0
_output_shapes
:�����������
3sequential/instance_normalization_2/batchnorm/RsqrtRsqrt5sequential/instance_normalization_2/batchnorm/add:z:0*
T0*0
_output_shapes
:�����������
1sequential/instance_normalization_2/batchnorm/mulMul7sequential/instance_normalization_2/batchnorm/Rsqrt:y:04sequential/instance_normalization_2/Reshape:output:0*
T0*0
_output_shapes
:�����������
3sequential/instance_normalization_2/batchnorm/mul_1Mul$sequential/conv2d_2/BiasAdd:output:05sequential/instance_normalization_2/batchnorm/mul:z:0*
T0*0
_output_shapes
:���������@@��
3sequential/instance_normalization_2/batchnorm/mul_2Mul9sequential/instance_normalization_2/moments/mean:output:05sequential/instance_normalization_2/batchnorm/mul:z:0*
T0*0
_output_shapes
:�����������
1sequential/instance_normalization_2/batchnorm/subSub6sequential/instance_normalization_2/Reshape_1:output:07sequential/instance_normalization_2/batchnorm/mul_2:z:0*
T0*0
_output_shapes
:�����������
3sequential/instance_normalization_2/batchnorm/add_1AddV27sequential/instance_normalization_2/batchnorm/mul_1:z:05sequential/instance_normalization_2/batchnorm/sub:z:0*
T0*0
_output_shapes
:���������@@��
sequential/re_lu_2/ReluRelu7sequential/instance_normalization_2/batchnorm/add_1:z:0*
T0*0
_output_shapes
:���������@@��
2sequential/residual/conv2d_3/Conv2D/ReadVariableOpReadVariableOp;sequential_residual_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
#sequential/residual/conv2d_3/Conv2DConv2D%sequential/re_lu_2/Relu:activations:0:sequential/residual/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
3sequential/residual/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp<sequential_residual_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$sequential/residual/conv2d_3/BiasAddBiasAdd,sequential/residual/conv2d_3/Conv2D:output:0;sequential/residual/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
!sequential/residual/conv2d_3/ReluRelu-sequential/residual/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
2sequential/residual/conv2d_4/Conv2D/ReadVariableOpReadVariableOp;sequential_residual_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
#sequential/residual/conv2d_4/Conv2DConv2D/sequential/residual/conv2d_3/Relu:activations:0:sequential/residual/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
3sequential/residual/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp<sequential_residual_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$sequential/residual/conv2d_4/BiasAddBiasAdd,sequential/residual/conv2d_4/Conv2D:output:0;sequential/residual/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
sequential/residual/add/addAddV2-sequential/residual/conv2d_4/BiasAdd:output:0%sequential/re_lu_2/Relu:activations:0*
T0*0
_output_shapes
:���������@@��
sequential/residual/re_lu/ReluRelusequential/residual/add/add:z:0*
T0*0
_output_shapes
:���������@@��
4sequential/residual_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp=sequential_residual_1_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
%sequential/residual_1/conv2d_5/Conv2DConv2D,sequential/residual/re_lu/Relu:activations:0<sequential/residual_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
5sequential/residual_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp>sequential_residual_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&sequential/residual_1/conv2d_5/BiasAddBiasAdd.sequential/residual_1/conv2d_5/Conv2D:output:0=sequential/residual_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
#sequential/residual_1/conv2d_5/ReluRelu/sequential/residual_1/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
4sequential/residual_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp=sequential_residual_1_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
%sequential/residual_1/conv2d_6/Conv2DConv2D1sequential/residual_1/conv2d_5/Relu:activations:0<sequential/residual_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
5sequential/residual_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp>sequential_residual_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&sequential/residual_1/conv2d_6/BiasAddBiasAdd.sequential/residual_1/conv2d_6/Conv2D:output:0=sequential/residual_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
sequential/residual_1/add_1/addAddV2/sequential/residual_1/conv2d_6/BiasAdd:output:0,sequential/residual/re_lu/Relu:activations:0*
T0*0
_output_shapes
:���������@@��
"sequential/residual_1/re_lu_1/ReluRelu#sequential/residual_1/add_1/add:z:0*
T0*0
_output_shapes
:���������@@��
4sequential/residual_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp=sequential_residual_2_conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
%sequential/residual_2/conv2d_7/Conv2DConv2D0sequential/residual_1/re_lu_1/Relu:activations:0<sequential/residual_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
5sequential/residual_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp>sequential_residual_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&sequential/residual_2/conv2d_7/BiasAddBiasAdd.sequential/residual_2/conv2d_7/Conv2D:output:0=sequential/residual_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
#sequential/residual_2/conv2d_7/ReluRelu/sequential/residual_2/conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
4sequential/residual_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp=sequential_residual_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
%sequential/residual_2/conv2d_8/Conv2DConv2D1sequential/residual_2/conv2d_7/Relu:activations:0<sequential/residual_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
5sequential/residual_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp>sequential_residual_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&sequential/residual_2/conv2d_8/BiasAddBiasAdd.sequential/residual_2/conv2d_8/Conv2D:output:0=sequential/residual_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
sequential/residual_2/add_2/addAddV2/sequential/residual_2/conv2d_8/BiasAdd:output:00sequential/residual_1/re_lu_1/Relu:activations:0*
T0*0
_output_shapes
:���������@@��
"sequential/residual_2/re_lu_2/ReluRelu#sequential/residual_2/add_2/add:z:0*
T0*0
_output_shapes
:���������@@��
4sequential/residual_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp=sequential_residual_3_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
%sequential/residual_3/conv2d_9/Conv2DConv2D0sequential/residual_2/re_lu_2/Relu:activations:0<sequential/residual_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
5sequential/residual_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp>sequential_residual_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&sequential/residual_3/conv2d_9/BiasAddBiasAdd.sequential/residual_3/conv2d_9/Conv2D:output:0=sequential/residual_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
#sequential/residual_3/conv2d_9/ReluRelu/sequential/residual_3/conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
5sequential/residual_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp>sequential_residual_3_conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&sequential/residual_3/conv2d_10/Conv2DConv2D1sequential/residual_3/conv2d_9/Relu:activations:0=sequential/residual_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
6sequential/residual_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp?sequential_residual_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'sequential/residual_3/conv2d_10/BiasAddBiasAdd/sequential/residual_3/conv2d_10/Conv2D:output:0>sequential/residual_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
sequential/residual_3/add_3/addAddV20sequential/residual_3/conv2d_10/BiasAdd:output:00sequential/residual_2/re_lu_2/Relu:activations:0*
T0*0
_output_shapes
:���������@@��
"sequential/residual_3/re_lu_3/ReluRelu#sequential/residual_3/add_3/add:z:0*
T0*0
_output_shapes
:���������@@��
5sequential/residual_4/conv2d_11/Conv2D/ReadVariableOpReadVariableOp>sequential_residual_4_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&sequential/residual_4/conv2d_11/Conv2DConv2D0sequential/residual_3/re_lu_3/Relu:activations:0=sequential/residual_4/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
6sequential/residual_4/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp?sequential_residual_4_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'sequential/residual_4/conv2d_11/BiasAddBiasAdd/sequential/residual_4/conv2d_11/Conv2D:output:0>sequential/residual_4/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
$sequential/residual_4/conv2d_11/ReluRelu0sequential/residual_4/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
5sequential/residual_4/conv2d_12/Conv2D/ReadVariableOpReadVariableOp>sequential_residual_4_conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&sequential/residual_4/conv2d_12/Conv2DConv2D2sequential/residual_4/conv2d_11/Relu:activations:0=sequential/residual_4/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
6sequential/residual_4/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp?sequential_residual_4_conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'sequential/residual_4/conv2d_12/BiasAddBiasAdd/sequential/residual_4/conv2d_12/Conv2D:output:0>sequential/residual_4/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
sequential/residual_4/add_4/addAddV20sequential/residual_4/conv2d_12/BiasAdd:output:00sequential/residual_3/re_lu_3/Relu:activations:0*
T0*0
_output_shapes
:���������@@��
"sequential/residual_4/re_lu_4/ReluRelu#sequential/residual_4/add_4/add:z:0*
T0*0
_output_shapes
:���������@@��
5sequential/residual_5/conv2d_13/Conv2D/ReadVariableOpReadVariableOp>sequential_residual_5_conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&sequential/residual_5/conv2d_13/Conv2DConv2D0sequential/residual_4/re_lu_4/Relu:activations:0=sequential/residual_5/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
6sequential/residual_5/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp?sequential_residual_5_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'sequential/residual_5/conv2d_13/BiasAddBiasAdd/sequential/residual_5/conv2d_13/Conv2D:output:0>sequential/residual_5/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
$sequential/residual_5/conv2d_13/ReluRelu0sequential/residual_5/conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
5sequential/residual_5/conv2d_14/Conv2D/ReadVariableOpReadVariableOp>sequential_residual_5_conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&sequential/residual_5/conv2d_14/Conv2DConv2D2sequential/residual_5/conv2d_13/Relu:activations:0=sequential/residual_5/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
6sequential/residual_5/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp?sequential_residual_5_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'sequential/residual_5/conv2d_14/BiasAddBiasAdd/sequential/residual_5/conv2d_14/Conv2D:output:0>sequential/residual_5/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
sequential/residual_5/add_5/addAddV20sequential/residual_5/conv2d_14/BiasAdd:output:00sequential/residual_4/re_lu_4/Relu:activations:0*
T0*0
_output_shapes
:���������@@��
"sequential/residual_5/re_lu_5/ReluRelu#sequential/residual_5/add_5/add:z:0*
T0*0
_output_shapes
:���������@@��
5sequential/residual_6/conv2d_15/Conv2D/ReadVariableOpReadVariableOp>sequential_residual_6_conv2d_15_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&sequential/residual_6/conv2d_15/Conv2DConv2D0sequential/residual_5/re_lu_5/Relu:activations:0=sequential/residual_6/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
6sequential/residual_6/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp?sequential_residual_6_conv2d_15_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'sequential/residual_6/conv2d_15/BiasAddBiasAdd/sequential/residual_6/conv2d_15/Conv2D:output:0>sequential/residual_6/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
$sequential/residual_6/conv2d_15/ReluRelu0sequential/residual_6/conv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
5sequential/residual_6/conv2d_16/Conv2D/ReadVariableOpReadVariableOp>sequential_residual_6_conv2d_16_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&sequential/residual_6/conv2d_16/Conv2DConv2D2sequential/residual_6/conv2d_15/Relu:activations:0=sequential/residual_6/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
6sequential/residual_6/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp?sequential_residual_6_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'sequential/residual_6/conv2d_16/BiasAddBiasAdd/sequential/residual_6/conv2d_16/Conv2D:output:0>sequential/residual_6/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
sequential/residual_6/add_6/addAddV20sequential/residual_6/conv2d_16/BiasAdd:output:00sequential/residual_5/re_lu_5/Relu:activations:0*
T0*0
_output_shapes
:���������@@��
"sequential/residual_6/re_lu_6/ReluRelu#sequential/residual_6/add_6/add:z:0*
T0*0
_output_shapes
:���������@@��
5sequential/residual_7/conv2d_17/Conv2D/ReadVariableOpReadVariableOp>sequential_residual_7_conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&sequential/residual_7/conv2d_17/Conv2DConv2D0sequential/residual_6/re_lu_6/Relu:activations:0=sequential/residual_7/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
6sequential/residual_7/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp?sequential_residual_7_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'sequential/residual_7/conv2d_17/BiasAddBiasAdd/sequential/residual_7/conv2d_17/Conv2D:output:0>sequential/residual_7/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
$sequential/residual_7/conv2d_17/ReluRelu0sequential/residual_7/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
5sequential/residual_7/conv2d_18/Conv2D/ReadVariableOpReadVariableOp>sequential_residual_7_conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&sequential/residual_7/conv2d_18/Conv2DConv2D2sequential/residual_7/conv2d_17/Relu:activations:0=sequential/residual_7/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
6sequential/residual_7/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp?sequential_residual_7_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'sequential/residual_7/conv2d_18/BiasAddBiasAdd/sequential/residual_7/conv2d_18/Conv2D:output:0>sequential/residual_7/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
sequential/residual_7/add_7/addAddV20sequential/residual_7/conv2d_18/BiasAdd:output:00sequential/residual_6/re_lu_6/Relu:activations:0*
T0*0
_output_shapes
:���������@@��
"sequential/residual_7/re_lu_7/ReluRelu#sequential/residual_7/add_7/add:z:0*
T0*0
_output_shapes
:���������@@��
5sequential/residual_8/conv2d_19/Conv2D/ReadVariableOpReadVariableOp>sequential_residual_8_conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&sequential/residual_8/conv2d_19/Conv2DConv2D0sequential/residual_7/re_lu_7/Relu:activations:0=sequential/residual_8/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
6sequential/residual_8/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp?sequential_residual_8_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'sequential/residual_8/conv2d_19/BiasAddBiasAdd/sequential/residual_8/conv2d_19/Conv2D:output:0>sequential/residual_8/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
$sequential/residual_8/conv2d_19/ReluRelu0sequential/residual_8/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
5sequential/residual_8/conv2d_20/Conv2D/ReadVariableOpReadVariableOp>sequential_residual_8_conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&sequential/residual_8/conv2d_20/Conv2DConv2D2sequential/residual_8/conv2d_19/Relu:activations:0=sequential/residual_8/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
6sequential/residual_8/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp?sequential_residual_8_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'sequential/residual_8/conv2d_20/BiasAddBiasAdd/sequential/residual_8/conv2d_20/Conv2D:output:0>sequential/residual_8/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
sequential/residual_8/add_8/addAddV20sequential/residual_8/conv2d_20/BiasAdd:output:00sequential/residual_7/re_lu_7/Relu:activations:0*
T0*0
_output_shapes
:���������@@��
"sequential/residual_8/re_lu_8/ReluRelu#sequential/residual_8/add_8/add:z:0*
T0*0
_output_shapes
:���������@@��
!sequential/conv2d_transpose/ShapeShape0sequential/residual_8/re_lu_8/Relu:activations:0*
T0*
_output_shapes
::��y
/sequential/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)sequential/conv2d_transpose/strided_sliceStridedSlice*sequential/conv2d_transpose/Shape:output:08sequential/conv2d_transpose/strided_slice/stack:output:0:sequential/conv2d_transpose/strided_slice/stack_1:output:0:sequential/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
#sequential/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�f
#sequential/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�f
#sequential/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
!sequential/conv2d_transpose/stackPack2sequential/conv2d_transpose/strided_slice:output:0,sequential/conv2d_transpose/stack/1:output:0,sequential/conv2d_transpose/stack/2:output:0,sequential/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:{
1sequential/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+sequential/conv2d_transpose/strided_slice_1StridedSlice*sequential/conv2d_transpose/stack:output:0:sequential/conv2d_transpose/strided_slice_1/stack:output:0<sequential/conv2d_transpose/strided_slice_1/stack_1:output:0<sequential/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpDsequential_conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
,sequential/conv2d_transpose/conv2d_transposeConv2DBackpropInput*sequential/conv2d_transpose/stack:output:0Csequential/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:00sequential/residual_8/re_lu_8/Relu:activations:0*
T0*2
_output_shapes 
:������������*
paddingSAME*
strides
�
2sequential/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp;sequential_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#sequential/conv2d_transpose/BiasAddBiasAdd5sequential/conv2d_transpose/conv2d_transpose:output:0:sequential/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:�������������
)sequential/instance_normalization_3/ShapeShape,sequential/conv2d_transpose/BiasAdd:output:0*
T0*
_output_shapes
::���
7sequential/instance_normalization_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
9sequential/instance_normalization_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9sequential/instance_normalization_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1sequential/instance_normalization_3/strided_sliceStridedSlice2sequential/instance_normalization_3/Shape:output:0@sequential/instance_normalization_3/strided_slice/stack:output:0Bsequential/instance_normalization_3/strided_slice/stack_1:output:0Bsequential/instance_normalization_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
9sequential/instance_normalization_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3sequential/instance_normalization_3/strided_slice_1StridedSlice2sequential/instance_normalization_3/Shape:output:0Bsequential/instance_normalization_3/strided_slice_1/stack:output:0Dsequential/instance_normalization_3/strided_slice_1/stack_1:output:0Dsequential/instance_normalization_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
9sequential/instance_normalization_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3sequential/instance_normalization_3/strided_slice_2StridedSlice2sequential/instance_normalization_3/Shape:output:0Bsequential/instance_normalization_3/strided_slice_2/stack:output:0Dsequential/instance_normalization_3/strided_slice_2/stack_1:output:0Dsequential/instance_normalization_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
9sequential/instance_normalization_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3sequential/instance_normalization_3/strided_slice_3StridedSlice2sequential/instance_normalization_3/Shape:output:0Bsequential/instance_normalization_3/strided_slice_3/stack:output:0Dsequential/instance_normalization_3/strided_slice_3/stack_1:output:0Dsequential/instance_normalization_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Bsequential/instance_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
0sequential/instance_normalization_3/moments/meanMean,sequential/conv2d_transpose/BiasAdd:output:0Ksequential/instance_normalization_3/moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
8sequential/instance_normalization_3/moments/StopGradientStopGradient9sequential/instance_normalization_3/moments/mean:output:0*
T0*0
_output_shapes
:�����������
=sequential/instance_normalization_3/moments/SquaredDifferenceSquaredDifference,sequential/conv2d_transpose/BiasAdd:output:0Asequential/instance_normalization_3/moments/StopGradient:output:0*
T0*2
_output_shapes 
:�������������
Fsequential/instance_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
4sequential/instance_normalization_3/moments/varianceMeanAsequential/instance_normalization_3/moments/SquaredDifference:z:0Osequential/instance_normalization_3/moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:����������*
	keep_dims(�
:sequential/instance_normalization_3/Reshape/ReadVariableOpReadVariableOpCsequential_instance_normalization_3_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
1sequential/instance_normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
+sequential/instance_normalization_3/ReshapeReshapeBsequential/instance_normalization_3/Reshape/ReadVariableOp:value:0:sequential/instance_normalization_3/Reshape/shape:output:0*
T0*'
_output_shapes
:��
<sequential/instance_normalization_3/Reshape_1/ReadVariableOpReadVariableOpEsequential_instance_normalization_3_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
3sequential/instance_normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
-sequential/instance_normalization_3/Reshape_1ReshapeDsequential/instance_normalization_3/Reshape_1/ReadVariableOp:value:0<sequential/instance_normalization_3/Reshape_1/shape:output:0*
T0*'
_output_shapes
:�x
3sequential/instance_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1sequential/instance_normalization_3/batchnorm/addAddV2=sequential/instance_normalization_3/moments/variance:output:0<sequential/instance_normalization_3/batchnorm/add/y:output:0*
T0*0
_output_shapes
:�����������
3sequential/instance_normalization_3/batchnorm/RsqrtRsqrt5sequential/instance_normalization_3/batchnorm/add:z:0*
T0*0
_output_shapes
:�����������
1sequential/instance_normalization_3/batchnorm/mulMul7sequential/instance_normalization_3/batchnorm/Rsqrt:y:04sequential/instance_normalization_3/Reshape:output:0*
T0*0
_output_shapes
:�����������
3sequential/instance_normalization_3/batchnorm/mul_1Mul,sequential/conv2d_transpose/BiasAdd:output:05sequential/instance_normalization_3/batchnorm/mul:z:0*
T0*2
_output_shapes 
:�������������
3sequential/instance_normalization_3/batchnorm/mul_2Mul9sequential/instance_normalization_3/moments/mean:output:05sequential/instance_normalization_3/batchnorm/mul:z:0*
T0*0
_output_shapes
:�����������
1sequential/instance_normalization_3/batchnorm/subSub6sequential/instance_normalization_3/Reshape_1:output:07sequential/instance_normalization_3/batchnorm/mul_2:z:0*
T0*0
_output_shapes
:�����������
3sequential/instance_normalization_3/batchnorm/add_1AddV27sequential/instance_normalization_3/batchnorm/mul_1:z:05sequential/instance_normalization_3/batchnorm/sub:z:0*
T0*2
_output_shapes 
:�������������
sequential/re_lu_3/ReluRelu7sequential/instance_normalization_3/batchnorm/add_1:z:0*
T0*2
_output_shapes 
:�������������
#sequential/conv2d_transpose_1/ShapeShape%sequential/re_lu_3/Relu:activations:0*
T0*
_output_shapes
::��{
1sequential/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+sequential/conv2d_transpose_1/strided_sliceStridedSlice,sequential/conv2d_transpose_1/Shape:output:0:sequential/conv2d_transpose_1/strided_slice/stack:output:0<sequential/conv2d_transpose_1/strided_slice/stack_1:output:0<sequential/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%sequential/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�h
%sequential/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�g
%sequential/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
#sequential/conv2d_transpose_1/stackPack4sequential/conv2d_transpose_1/strided_slice:output:0.sequential/conv2d_transpose_1/stack/1:output:0.sequential/conv2d_transpose_1/stack/2:output:0.sequential/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:}
3sequential/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-sequential/conv2d_transpose_1/strided_slice_1StridedSlice,sequential/conv2d_transpose_1/stack:output:0<sequential/conv2d_transpose_1/strided_slice_1/stack:output:0>sequential/conv2d_transpose_1/strided_slice_1/stack_1:output:0>sequential/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
.sequential/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput,sequential/conv2d_transpose_1/stack:output:0Esequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0%sequential/re_lu_3/Relu:activations:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
4sequential/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp=sequential_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
%sequential/conv2d_transpose_1/BiasAddBiasAdd7sequential/conv2d_transpose_1/conv2d_transpose:output:0<sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@�
)sequential/instance_normalization_4/ShapeShape.sequential/conv2d_transpose_1/BiasAdd:output:0*
T0*
_output_shapes
::���
7sequential/instance_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
9sequential/instance_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9sequential/instance_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1sequential/instance_normalization_4/strided_sliceStridedSlice2sequential/instance_normalization_4/Shape:output:0@sequential/instance_normalization_4/strided_slice/stack:output:0Bsequential/instance_normalization_4/strided_slice/stack_1:output:0Bsequential/instance_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
9sequential/instance_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3sequential/instance_normalization_4/strided_slice_1StridedSlice2sequential/instance_normalization_4/Shape:output:0Bsequential/instance_normalization_4/strided_slice_1/stack:output:0Dsequential/instance_normalization_4/strided_slice_1/stack_1:output:0Dsequential/instance_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
9sequential/instance_normalization_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3sequential/instance_normalization_4/strided_slice_2StridedSlice2sequential/instance_normalization_4/Shape:output:0Bsequential/instance_normalization_4/strided_slice_2/stack:output:0Dsequential/instance_normalization_4/strided_slice_2/stack_1:output:0Dsequential/instance_normalization_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
9sequential/instance_normalization_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;sequential/instance_normalization_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3sequential/instance_normalization_4/strided_slice_3StridedSlice2sequential/instance_normalization_4/Shape:output:0Bsequential/instance_normalization_4/strided_slice_3/stack:output:0Dsequential/instance_normalization_4/strided_slice_3/stack_1:output:0Dsequential/instance_normalization_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Bsequential/instance_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
0sequential/instance_normalization_4/moments/meanMean.sequential/conv2d_transpose_1/BiasAdd:output:0Ksequential/instance_normalization_4/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������@*
	keep_dims(�
8sequential/instance_normalization_4/moments/StopGradientStopGradient9sequential/instance_normalization_4/moments/mean:output:0*
T0*/
_output_shapes
:���������@�
=sequential/instance_normalization_4/moments/SquaredDifferenceSquaredDifference.sequential/conv2d_transpose_1/BiasAdd:output:0Asequential/instance_normalization_4/moments/StopGradient:output:0*
T0*1
_output_shapes
:�����������@�
Fsequential/instance_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
4sequential/instance_normalization_4/moments/varianceMeanAsequential/instance_normalization_4/moments/SquaredDifference:z:0Osequential/instance_normalization_4/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:���������@*
	keep_dims(�
:sequential/instance_normalization_4/Reshape/ReadVariableOpReadVariableOpCsequential_instance_normalization_4_reshape_readvariableop_resource*
_output_shapes
:@*
dtype0�
1sequential/instance_normalization_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
+sequential/instance_normalization_4/ReshapeReshapeBsequential/instance_normalization_4/Reshape/ReadVariableOp:value:0:sequential/instance_normalization_4/Reshape/shape:output:0*
T0*&
_output_shapes
:@�
<sequential/instance_normalization_4/Reshape_1/ReadVariableOpReadVariableOpEsequential_instance_normalization_4_reshape_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
3sequential/instance_normalization_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
-sequential/instance_normalization_4/Reshape_1ReshapeDsequential/instance_normalization_4/Reshape_1/ReadVariableOp:value:0<sequential/instance_normalization_4/Reshape_1/shape:output:0*
T0*&
_output_shapes
:@x
3sequential/instance_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1sequential/instance_normalization_4/batchnorm/addAddV2=sequential/instance_normalization_4/moments/variance:output:0<sequential/instance_normalization_4/batchnorm/add/y:output:0*
T0*/
_output_shapes
:���������@�
3sequential/instance_normalization_4/batchnorm/RsqrtRsqrt5sequential/instance_normalization_4/batchnorm/add:z:0*
T0*/
_output_shapes
:���������@�
1sequential/instance_normalization_4/batchnorm/mulMul7sequential/instance_normalization_4/batchnorm/Rsqrt:y:04sequential/instance_normalization_4/Reshape:output:0*
T0*/
_output_shapes
:���������@�
3sequential/instance_normalization_4/batchnorm/mul_1Mul.sequential/conv2d_transpose_1/BiasAdd:output:05sequential/instance_normalization_4/batchnorm/mul:z:0*
T0*1
_output_shapes
:�����������@�
3sequential/instance_normalization_4/batchnorm/mul_2Mul9sequential/instance_normalization_4/moments/mean:output:05sequential/instance_normalization_4/batchnorm/mul:z:0*
T0*/
_output_shapes
:���������@�
1sequential/instance_normalization_4/batchnorm/subSub6sequential/instance_normalization_4/Reshape_1:output:07sequential/instance_normalization_4/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:���������@�
3sequential/instance_normalization_4/batchnorm/add_1AddV27sequential/instance_normalization_4/batchnorm/mul_1:z:05sequential/instance_normalization_4/batchnorm/sub:z:0*
T0*1
_output_shapes
:�����������@�
sequential/re_lu_4/ReluRelu7sequential/instance_normalization_4/batchnorm/add_1:z:0*
T0*1
_output_shapes
:�����������@�
*sequential/conv2d_21/Conv2D/ReadVariableOpReadVariableOp3sequential_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
sequential/conv2d_21/Conv2DConv2D%sequential/re_lu_4/Relu:activations:02sequential/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
+sequential/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp4sequential_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/conv2d_21/BiasAddBiasAdd$sequential/conv2d_21/Conv2D:output:03sequential/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
sequential/conv2d_21/TanhTanh%sequential/conv2d_21/BiasAdd:output:0*
T0*1
_output_shapes
:�����������v
IdentityIdentitysequential/conv2d_21/Tanh:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp,^sequential/conv2d_21/BiasAdd/ReadVariableOp+^sequential/conv2d_21/Conv2D/ReadVariableOp3^sequential/conv2d_transpose/BiasAdd/ReadVariableOp<^sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp5^sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp>^sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp9^sequential/instance_normalization/Reshape/ReadVariableOp;^sequential/instance_normalization/Reshape_1/ReadVariableOp;^sequential/instance_normalization_1/Reshape/ReadVariableOp=^sequential/instance_normalization_1/Reshape_1/ReadVariableOp;^sequential/instance_normalization_2/Reshape/ReadVariableOp=^sequential/instance_normalization_2/Reshape_1/ReadVariableOp;^sequential/instance_normalization_3/Reshape/ReadVariableOp=^sequential/instance_normalization_3/Reshape_1/ReadVariableOp;^sequential/instance_normalization_4/Reshape/ReadVariableOp=^sequential/instance_normalization_4/Reshape_1/ReadVariableOp4^sequential/residual/conv2d_3/BiasAdd/ReadVariableOp3^sequential/residual/conv2d_3/Conv2D/ReadVariableOp4^sequential/residual/conv2d_4/BiasAdd/ReadVariableOp3^sequential/residual/conv2d_4/Conv2D/ReadVariableOp6^sequential/residual_1/conv2d_5/BiasAdd/ReadVariableOp5^sequential/residual_1/conv2d_5/Conv2D/ReadVariableOp6^sequential/residual_1/conv2d_6/BiasAdd/ReadVariableOp5^sequential/residual_1/conv2d_6/Conv2D/ReadVariableOp6^sequential/residual_2/conv2d_7/BiasAdd/ReadVariableOp5^sequential/residual_2/conv2d_7/Conv2D/ReadVariableOp6^sequential/residual_2/conv2d_8/BiasAdd/ReadVariableOp5^sequential/residual_2/conv2d_8/Conv2D/ReadVariableOp7^sequential/residual_3/conv2d_10/BiasAdd/ReadVariableOp6^sequential/residual_3/conv2d_10/Conv2D/ReadVariableOp6^sequential/residual_3/conv2d_9/BiasAdd/ReadVariableOp5^sequential/residual_3/conv2d_9/Conv2D/ReadVariableOp7^sequential/residual_4/conv2d_11/BiasAdd/ReadVariableOp6^sequential/residual_4/conv2d_11/Conv2D/ReadVariableOp7^sequential/residual_4/conv2d_12/BiasAdd/ReadVariableOp6^sequential/residual_4/conv2d_12/Conv2D/ReadVariableOp7^sequential/residual_5/conv2d_13/BiasAdd/ReadVariableOp6^sequential/residual_5/conv2d_13/Conv2D/ReadVariableOp7^sequential/residual_5/conv2d_14/BiasAdd/ReadVariableOp6^sequential/residual_5/conv2d_14/Conv2D/ReadVariableOp7^sequential/residual_6/conv2d_15/BiasAdd/ReadVariableOp6^sequential/residual_6/conv2d_15/Conv2D/ReadVariableOp7^sequential/residual_6/conv2d_16/BiasAdd/ReadVariableOp6^sequential/residual_6/conv2d_16/Conv2D/ReadVariableOp7^sequential/residual_7/conv2d_17/BiasAdd/ReadVariableOp6^sequential/residual_7/conv2d_17/Conv2D/ReadVariableOp7^sequential/residual_7/conv2d_18/BiasAdd/ReadVariableOp6^sequential/residual_7/conv2d_18/Conv2D/ReadVariableOp7^sequential/residual_8/conv2d_19/BiasAdd/ReadVariableOp6^sequential/residual_8/conv2d_19/Conv2D/ReadVariableOp7^sequential/residual_8/conv2d_20/BiasAdd/ReadVariableOp6^sequential/residual_8/conv2d_20/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2Z
+sequential/conv2d_21/BiasAdd/ReadVariableOp+sequential/conv2d_21/BiasAdd/ReadVariableOp2X
*sequential/conv2d_21/Conv2D/ReadVariableOp*sequential/conv2d_21/Conv2D/ReadVariableOp2h
2sequential/conv2d_transpose/BiasAdd/ReadVariableOp2sequential/conv2d_transpose/BiasAdd/ReadVariableOp2z
;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp2l
4sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp4sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp2~
=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2t
8sequential/instance_normalization/Reshape/ReadVariableOp8sequential/instance_normalization/Reshape/ReadVariableOp2x
:sequential/instance_normalization/Reshape_1/ReadVariableOp:sequential/instance_normalization/Reshape_1/ReadVariableOp2x
:sequential/instance_normalization_1/Reshape/ReadVariableOp:sequential/instance_normalization_1/Reshape/ReadVariableOp2|
<sequential/instance_normalization_1/Reshape_1/ReadVariableOp<sequential/instance_normalization_1/Reshape_1/ReadVariableOp2x
:sequential/instance_normalization_2/Reshape/ReadVariableOp:sequential/instance_normalization_2/Reshape/ReadVariableOp2|
<sequential/instance_normalization_2/Reshape_1/ReadVariableOp<sequential/instance_normalization_2/Reshape_1/ReadVariableOp2x
:sequential/instance_normalization_3/Reshape/ReadVariableOp:sequential/instance_normalization_3/Reshape/ReadVariableOp2|
<sequential/instance_normalization_3/Reshape_1/ReadVariableOp<sequential/instance_normalization_3/Reshape_1/ReadVariableOp2x
:sequential/instance_normalization_4/Reshape/ReadVariableOp:sequential/instance_normalization_4/Reshape/ReadVariableOp2|
<sequential/instance_normalization_4/Reshape_1/ReadVariableOp<sequential/instance_normalization_4/Reshape_1/ReadVariableOp2j
3sequential/residual/conv2d_3/BiasAdd/ReadVariableOp3sequential/residual/conv2d_3/BiasAdd/ReadVariableOp2h
2sequential/residual/conv2d_3/Conv2D/ReadVariableOp2sequential/residual/conv2d_3/Conv2D/ReadVariableOp2j
3sequential/residual/conv2d_4/BiasAdd/ReadVariableOp3sequential/residual/conv2d_4/BiasAdd/ReadVariableOp2h
2sequential/residual/conv2d_4/Conv2D/ReadVariableOp2sequential/residual/conv2d_4/Conv2D/ReadVariableOp2n
5sequential/residual_1/conv2d_5/BiasAdd/ReadVariableOp5sequential/residual_1/conv2d_5/BiasAdd/ReadVariableOp2l
4sequential/residual_1/conv2d_5/Conv2D/ReadVariableOp4sequential/residual_1/conv2d_5/Conv2D/ReadVariableOp2n
5sequential/residual_1/conv2d_6/BiasAdd/ReadVariableOp5sequential/residual_1/conv2d_6/BiasAdd/ReadVariableOp2l
4sequential/residual_1/conv2d_6/Conv2D/ReadVariableOp4sequential/residual_1/conv2d_6/Conv2D/ReadVariableOp2n
5sequential/residual_2/conv2d_7/BiasAdd/ReadVariableOp5sequential/residual_2/conv2d_7/BiasAdd/ReadVariableOp2l
4sequential/residual_2/conv2d_7/Conv2D/ReadVariableOp4sequential/residual_2/conv2d_7/Conv2D/ReadVariableOp2n
5sequential/residual_2/conv2d_8/BiasAdd/ReadVariableOp5sequential/residual_2/conv2d_8/BiasAdd/ReadVariableOp2l
4sequential/residual_2/conv2d_8/Conv2D/ReadVariableOp4sequential/residual_2/conv2d_8/Conv2D/ReadVariableOp2p
6sequential/residual_3/conv2d_10/BiasAdd/ReadVariableOp6sequential/residual_3/conv2d_10/BiasAdd/ReadVariableOp2n
5sequential/residual_3/conv2d_10/Conv2D/ReadVariableOp5sequential/residual_3/conv2d_10/Conv2D/ReadVariableOp2n
5sequential/residual_3/conv2d_9/BiasAdd/ReadVariableOp5sequential/residual_3/conv2d_9/BiasAdd/ReadVariableOp2l
4sequential/residual_3/conv2d_9/Conv2D/ReadVariableOp4sequential/residual_3/conv2d_9/Conv2D/ReadVariableOp2p
6sequential/residual_4/conv2d_11/BiasAdd/ReadVariableOp6sequential/residual_4/conv2d_11/BiasAdd/ReadVariableOp2n
5sequential/residual_4/conv2d_11/Conv2D/ReadVariableOp5sequential/residual_4/conv2d_11/Conv2D/ReadVariableOp2p
6sequential/residual_4/conv2d_12/BiasAdd/ReadVariableOp6sequential/residual_4/conv2d_12/BiasAdd/ReadVariableOp2n
5sequential/residual_4/conv2d_12/Conv2D/ReadVariableOp5sequential/residual_4/conv2d_12/Conv2D/ReadVariableOp2p
6sequential/residual_5/conv2d_13/BiasAdd/ReadVariableOp6sequential/residual_5/conv2d_13/BiasAdd/ReadVariableOp2n
5sequential/residual_5/conv2d_13/Conv2D/ReadVariableOp5sequential/residual_5/conv2d_13/Conv2D/ReadVariableOp2p
6sequential/residual_5/conv2d_14/BiasAdd/ReadVariableOp6sequential/residual_5/conv2d_14/BiasAdd/ReadVariableOp2n
5sequential/residual_5/conv2d_14/Conv2D/ReadVariableOp5sequential/residual_5/conv2d_14/Conv2D/ReadVariableOp2p
6sequential/residual_6/conv2d_15/BiasAdd/ReadVariableOp6sequential/residual_6/conv2d_15/BiasAdd/ReadVariableOp2n
5sequential/residual_6/conv2d_15/Conv2D/ReadVariableOp5sequential/residual_6/conv2d_15/Conv2D/ReadVariableOp2p
6sequential/residual_6/conv2d_16/BiasAdd/ReadVariableOp6sequential/residual_6/conv2d_16/BiasAdd/ReadVariableOp2n
5sequential/residual_6/conv2d_16/Conv2D/ReadVariableOp5sequential/residual_6/conv2d_16/Conv2D/ReadVariableOp2p
6sequential/residual_7/conv2d_17/BiasAdd/ReadVariableOp6sequential/residual_7/conv2d_17/BiasAdd/ReadVariableOp2n
5sequential/residual_7/conv2d_17/Conv2D/ReadVariableOp5sequential/residual_7/conv2d_17/Conv2D/ReadVariableOp2p
6sequential/residual_7/conv2d_18/BiasAdd/ReadVariableOp6sequential/residual_7/conv2d_18/BiasAdd/ReadVariableOp2n
5sequential/residual_7/conv2d_18/Conv2D/ReadVariableOp5sequential/residual_7/conv2d_18/Conv2D/ReadVariableOp2p
6sequential/residual_8/conv2d_19/BiasAdd/ReadVariableOp6sequential/residual_8/conv2d_19/BiasAdd/ReadVariableOp2n
5sequential/residual_8/conv2d_19/Conv2D/ReadVariableOp5sequential/residual_8/conv2d_19/Conv2D/ReadVariableOp2p
6sequential/residual_8/conv2d_20/BiasAdd/ReadVariableOp6sequential/residual_8/conv2d_20/BiasAdd/ReadVariableOp2n
5sequential/residual_8/conv2d_20/Conv2D/ReadVariableOp5sequential/residual_8/conv2d_20/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�

�
E__inference_conv2d_16_layer_call_and_return_conditional_losses_745113

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
D__inference_conv2d_7_layer_call_and_return_conditional_losses_740840

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������@@�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
3__inference_conv2d_transpose_1_layer_call_fn_744725

inputs"
unknown:@�
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_741421�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
1__inference_conv2d_transpose_layer_call_fn_744621

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_741377�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
*__inference_conv2d_21_layer_call_fn_744829

inputs!
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2 *0,1,2J 8� *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_741865y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
F__inference_residual_1_layer_call_and_return_conditional_losses_744388

inputsC
'conv2d_5_conv2d_readvariableop_resource:��7
(conv2d_5_biasadd_readvariableop_resource:	�C
'conv2d_6_conv2d_readvariableop_resource:��7
(conv2d_6_biasadd_readvariableop_resource:	�
identity��conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�k
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:���������@@��
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_6/Conv2DConv2Dconv2d_5/Relu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�*
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@�n
add/addAddV2conv2d_6/BiasAdd:output:0inputs*
T0*0
_output_shapes
:���������@@�Z

re_lu/ReluReluadd/add:z:0*
T0*0
_output_shapes
:���������@@�p
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*0
_output_shapes
:���������@@��
NoOpNoOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������@@�: : : : 2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
E
input_1:
serving_default_input_1:0�����������G
	conv2d_21:
StatefulPartitionedCall:0�����������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer_with_weights-10
layer-13
layer_with_weights-11
layer-14
layer_with_weights-12
layer-15
layer_with_weights-13
layer-16
layer_with_weights-14
layer-17
layer_with_weights-15
layer-18
layer_with_weights-16
layer-19
layer-20
layer_with_weights-17
layer-21
layer_with_weights-18
layer-22
layer-23
layer_with_weights-19
layer-24
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _default_save_signature
!
signatures"
_tf_keras_sequential
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias
 *_jit_compiled_convolution_op"
_tf_keras_layer
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
	1gamma
2beta"
_tf_keras_layer
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias
 A_jit_compiled_convolution_op"
_tf_keras_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
	Hgamma
Ibeta"
_tf_keras_layer
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

Vkernel
Wbias
 X_jit_compiled_convolution_op"
_tf_keras_layer
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
	_gamma
`beta"
_tf_keras_layer
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses
	mconv1
	nconv2"
_tf_keras_model
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses
	uconv1
	vconv2"
_tf_keras_model
�
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses
	}conv1
	~conv2"
_tf_keras_model
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�conv1

�conv2"
_tf_keras_model
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�conv1

�conv2"
_tf_keras_model
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�conv1

�conv2"
_tf_keras_model
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�conv1

�conv2"
_tf_keras_model
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�conv1

�conv2"
_tf_keras_model
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�conv1

�conv2"
_tf_keras_model
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�gamma
	�beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�gamma
	�beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
(0
)1
12
23
?4
@5
H6
I7
V8
W9
_10
`11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57"
trackable_list_wrapper
�
(0
)1
12
23
?4
@5
H6
I7
V8
W9
_10
`11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
 _default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
+__inference_sequential_layer_call_fn_742284
+__inference_sequential_layer_call_fn_742550
+__inference_sequential_layer_call_fn_743156
+__inference_sequential_layer_call_fn_743277�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
F__inference_sequential_layer_call_and_return_conditional_losses_741872
F__inference_sequential_layer_call_and_return_conditional_losses_742017
F__inference_sequential_layer_call_and_return_conditional_losses_743679
F__inference_sequential_layer_call_and_return_conditional_losses_744081�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
!__inference__wrapped_model_740677input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
-
�serving_default"
signature_map
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv2d_layer_call_fn_744090�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_conv2d_layer_call_and_return_conditional_losses_744100�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
':%@2conv2d/kernel
:@2conv2d/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
7__inference_instance_normalization_layer_call_fn_744109�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
R__inference_instance_normalization_layer_call_and_return_conditional_losses_744152�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(@2instance_normalization/gamma
):'@2instance_normalization/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_re_lu_layer_call_fn_744157�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_re_lu_layer_call_and_return_conditional_losses_744162�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_1_layer_call_fn_744171�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_744181�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(@�2conv2d_1/kernel
:�2conv2d_1/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
9__inference_instance_normalization_1_layer_call_fn_744190�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
T__inference_instance_normalization_1_layer_call_and_return_conditional_losses_744233�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+�2instance_normalization_1/gamma
,:*�2instance_normalization_1/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_re_lu_1_layer_call_fn_744238�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_re_lu_1_layer_call_and_return_conditional_losses_744243�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_2_layer_call_fn_744252�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_744262�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:)��2conv2d_2/kernel
:�2conv2d_2/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
_0
`1"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
9__inference_instance_normalization_2_layer_call_fn_744271�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
T__inference_instance_normalization_2_layer_call_and_return_conditional_losses_744314�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+�2instance_normalization_2/gamma
,:*�2instance_normalization_2/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_re_lu_2_layer_call_fn_744319�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_re_lu_2_layer_call_and_return_conditional_losses_744324�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
@
�0
�1
�2
�3"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_residual_layer_call_fn_744337�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_residual_layer_call_and_return_conditional_losses_744356�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
@
�0
�1
�2
�3"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_residual_1_layer_call_fn_744369�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_residual_1_layer_call_and_return_conditional_losses_744388�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
@
�0
�1
�2
�3"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_residual_2_layer_call_fn_744401�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_residual_2_layer_call_and_return_conditional_losses_744420�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
@
�0
�1
�2
�3"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_residual_3_layer_call_fn_744433�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_residual_3_layer_call_and_return_conditional_losses_744452�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
@
�0
�1
�2
�3"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_residual_4_layer_call_fn_744465�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_residual_4_layer_call_and_return_conditional_losses_744484�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
@
�0
�1
�2
�3"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_residual_5_layer_call_fn_744497�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_residual_5_layer_call_and_return_conditional_losses_744516�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
@
�0
�1
�2
�3"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_residual_6_layer_call_fn_744529�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_residual_6_layer_call_and_return_conditional_losses_744548�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
@
�0
�1
�2
�3"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_residual_7_layer_call_fn_744561�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_residual_7_layer_call_and_return_conditional_losses_744580�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
@
�0
�1
�2
�3"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_residual_8_layer_call_fn_744593�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_residual_8_layer_call_and_return_conditional_losses_744612�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_conv2d_transpose_layer_call_fn_744621�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_744654�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
3:1��2conv2d_transpose/kernel
$:"�2conv2d_transpose/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
9__inference_instance_normalization_3_layer_call_fn_744663�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
T__inference_instance_normalization_3_layer_call_and_return_conditional_losses_744706�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+�2instance_normalization_3/gamma
,:*�2instance_normalization_3/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_re_lu_3_layer_call_fn_744711�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_re_lu_3_layer_call_and_return_conditional_losses_744716�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_conv2d_transpose_1_layer_call_fn_744725�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_744758�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
4:2@�2conv2d_transpose_1/kernel
%:#@2conv2d_transpose_1/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
9__inference_instance_normalization_4_layer_call_fn_744767�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
T__inference_instance_normalization_4_layer_call_and_return_conditional_losses_744810�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
,:*@2instance_normalization_4/gamma
+:)@2instance_normalization_4/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_re_lu_4_layer_call_fn_744815�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_re_lu_4_layer_call_and_return_conditional_losses_744820�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_21_layer_call_fn_744829�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_21_layer_call_and_return_conditional_losses_744840�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(@2conv2d_21/kernel
:2conv2d_21/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
4:2��2residual/conv2d_3/kernel
%:#�2residual/conv2d_3/bias
4:2��2residual/conv2d_4/kernel
%:#�2residual/conv2d_4/bias
6:4��2residual_1/conv2d_5/kernel
':%�2residual_1/conv2d_5/bias
6:4��2residual_1/conv2d_6/kernel
':%�2residual_1/conv2d_6/bias
6:4��2residual_2/conv2d_7/kernel
':%�2residual_2/conv2d_7/bias
6:4��2residual_2/conv2d_8/kernel
':%�2residual_2/conv2d_8/bias
6:4��2residual_3/conv2d_9/kernel
':%�2residual_3/conv2d_9/bias
7:5��2residual_3/conv2d_10/kernel
(:&�2residual_3/conv2d_10/bias
7:5��2residual_4/conv2d_11/kernel
(:&�2residual_4/conv2d_11/bias
7:5��2residual_4/conv2d_12/kernel
(:&�2residual_4/conv2d_12/bias
7:5��2residual_5/conv2d_13/kernel
(:&�2residual_5/conv2d_13/bias
7:5��2residual_5/conv2d_14/kernel
(:&�2residual_5/conv2d_14/bias
7:5��2residual_6/conv2d_15/kernel
(:&�2residual_6/conv2d_15/bias
7:5��2residual_6/conv2d_16/kernel
(:&�2residual_6/conv2d_16/bias
7:5��2residual_7/conv2d_17/kernel
(:&�2residual_7/conv2d_17/bias
7:5��2residual_7/conv2d_18/kernel
(:&�2residual_7/conv2d_18/bias
7:5��2residual_8/conv2d_19/kernel
(:&�2residual_8/conv2d_19/bias
7:5��2residual_8/conv2d_20/kernel
(:&�2residual_8/conv2d_20/bias
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_sequential_layer_call_fn_742284input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_sequential_layer_call_fn_742550input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_sequential_layer_call_fn_743156inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_sequential_layer_call_fn_743277inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_sequential_layer_call_and_return_conditional_losses_741872input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_sequential_layer_call_and_return_conditional_losses_742017input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_sequential_layer_call_and_return_conditional_losses_743679inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_sequential_layer_call_and_return_conditional_losses_744081inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_743035input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
'__inference_conv2d_layer_call_fn_744090inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_conv2d_layer_call_and_return_conditional_losses_744100inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
7__inference_instance_normalization_layer_call_fn_744109inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_instance_normalization_layer_call_and_return_conditional_losses_744152inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
&__inference_re_lu_layer_call_fn_744157inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_re_lu_layer_call_and_return_conditional_losses_744162inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_conv2d_1_layer_call_fn_744171inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_744181inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
9__inference_instance_normalization_1_layer_call_fn_744190inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_instance_normalization_1_layer_call_and_return_conditional_losses_744233inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
(__inference_re_lu_1_layer_call_fn_744238inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_re_lu_1_layer_call_and_return_conditional_losses_744243inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_conv2d_2_layer_call_fn_744252inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_744262inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
9__inference_instance_normalization_2_layer_call_fn_744271inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_instance_normalization_2_layer_call_and_return_conditional_losses_744314inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
(__inference_re_lu_2_layer_call_fn_744319inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_re_lu_2_layer_call_and_return_conditional_losses_744324inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_residual_layer_call_fn_744337inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_residual_layer_call_and_return_conditional_losses_744356inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_3_layer_call_fn_744849�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_744860�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_4_layer_call_fn_744869�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_4_layer_call_and_return_conditional_losses_744879�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_residual_1_layer_call_fn_744369inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_residual_1_layer_call_and_return_conditional_losses_744388inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_5_layer_call_fn_744888�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_5_layer_call_and_return_conditional_losses_744899�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_6_layer_call_fn_744908�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_6_layer_call_and_return_conditional_losses_744918�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_residual_2_layer_call_fn_744401inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_residual_2_layer_call_and_return_conditional_losses_744420inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_7_layer_call_fn_744927�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_7_layer_call_and_return_conditional_losses_744938�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_8_layer_call_fn_744947�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_8_layer_call_and_return_conditional_losses_744957�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_residual_3_layer_call_fn_744433inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_residual_3_layer_call_and_return_conditional_losses_744452inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_9_layer_call_fn_744966�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_9_layer_call_and_return_conditional_losses_744977�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_10_layer_call_fn_744986�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_10_layer_call_and_return_conditional_losses_744996�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_residual_4_layer_call_fn_744465inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_residual_4_layer_call_and_return_conditional_losses_744484inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_11_layer_call_fn_745005�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_11_layer_call_and_return_conditional_losses_745016�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_12_layer_call_fn_745025�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_12_layer_call_and_return_conditional_losses_745035�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_residual_5_layer_call_fn_744497inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_residual_5_layer_call_and_return_conditional_losses_744516inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_13_layer_call_fn_745044�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_13_layer_call_and_return_conditional_losses_745055�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_14_layer_call_fn_745064�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_14_layer_call_and_return_conditional_losses_745074�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_residual_6_layer_call_fn_744529inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_residual_6_layer_call_and_return_conditional_losses_744548inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_15_layer_call_fn_745083�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_15_layer_call_and_return_conditional_losses_745094�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_16_layer_call_fn_745103�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_16_layer_call_and_return_conditional_losses_745113�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_residual_7_layer_call_fn_744561inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_residual_7_layer_call_and_return_conditional_losses_744580inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_17_layer_call_fn_745122�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_17_layer_call_and_return_conditional_losses_745133�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_18_layer_call_fn_745142�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_18_layer_call_and_return_conditional_losses_745152�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_residual_8_layer_call_fn_744593inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_residual_8_layer_call_and_return_conditional_losses_744612inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_19_layer_call_fn_745161�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_19_layer_call_and_return_conditional_losses_745172�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_20_layer_call_fn_745181�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_20_layer_call_and_return_conditional_losses_745191�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
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
�B�
1__inference_conv2d_transpose_layer_call_fn_744621inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_744654inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
9__inference_instance_normalization_3_layer_call_fn_744663inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_instance_normalization_3_layer_call_and_return_conditional_losses_744706inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
(__inference_re_lu_3_layer_call_fn_744711inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_re_lu_3_layer_call_and_return_conditional_losses_744716inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
3__inference_conv2d_transpose_1_layer_call_fn_744725inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_744758inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
9__inference_instance_normalization_4_layer_call_fn_744767inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_instance_normalization_4_layer_call_and_return_conditional_losses_744810inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
(__inference_re_lu_4_layer_call_fn_744815inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_re_lu_4_layer_call_and_return_conditional_losses_744820inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_conv2d_21_layer_call_fn_744829inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_21_layer_call_and_return_conditional_losses_744840inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_conv2d_3_layer_call_fn_744849inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_744860inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_conv2d_4_layer_call_fn_744869inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_4_layer_call_and_return_conditional_losses_744879inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_conv2d_5_layer_call_fn_744888inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_5_layer_call_and_return_conditional_losses_744899inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_conv2d_6_layer_call_fn_744908inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_6_layer_call_and_return_conditional_losses_744918inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_conv2d_7_layer_call_fn_744927inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_7_layer_call_and_return_conditional_losses_744938inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_conv2d_8_layer_call_fn_744947inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_8_layer_call_and_return_conditional_losses_744957inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_conv2d_9_layer_call_fn_744966inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_9_layer_call_and_return_conditional_losses_744977inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_conv2d_10_layer_call_fn_744986inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_10_layer_call_and_return_conditional_losses_744996inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_conv2d_11_layer_call_fn_745005inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_11_layer_call_and_return_conditional_losses_745016inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_conv2d_12_layer_call_fn_745025inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_12_layer_call_and_return_conditional_losses_745035inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_conv2d_13_layer_call_fn_745044inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_13_layer_call_and_return_conditional_losses_745055inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_conv2d_14_layer_call_fn_745064inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_14_layer_call_and_return_conditional_losses_745074inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_conv2d_15_layer_call_fn_745083inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_15_layer_call_and_return_conditional_losses_745094inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_conv2d_16_layer_call_fn_745103inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_16_layer_call_and_return_conditional_losses_745113inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_conv2d_17_layer_call_fn_745122inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_17_layer_call_and_return_conditional_losses_745133inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_conv2d_18_layer_call_fn_745142inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_18_layer_call_and_return_conditional_losses_745152inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_conv2d_19_layer_call_fn_745161inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_19_layer_call_and_return_conditional_losses_745172inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_conv2d_20_layer_call_fn_745181inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_20_layer_call_and_return_conditional_losses_745191inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_740677�h()12?@HIVW_`����������������������������������������������:�7
0�-
+�(
input_1�����������
� "?�<
:
	conv2d_21-�*
	conv2d_21������������
E__inference_conv2d_10_layer_call_and_return_conditional_losses_744996w��8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
*__inference_conv2d_10_layer_call_fn_744986l��8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
E__inference_conv2d_11_layer_call_and_return_conditional_losses_745016w��8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
*__inference_conv2d_11_layer_call_fn_745005l��8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
E__inference_conv2d_12_layer_call_and_return_conditional_losses_745035w��8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
*__inference_conv2d_12_layer_call_fn_745025l��8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
E__inference_conv2d_13_layer_call_and_return_conditional_losses_745055w��8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
*__inference_conv2d_13_layer_call_fn_745044l��8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
E__inference_conv2d_14_layer_call_and_return_conditional_losses_745074w��8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
*__inference_conv2d_14_layer_call_fn_745064l��8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
E__inference_conv2d_15_layer_call_and_return_conditional_losses_745094w��8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
*__inference_conv2d_15_layer_call_fn_745083l��8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
E__inference_conv2d_16_layer_call_and_return_conditional_losses_745113w��8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
*__inference_conv2d_16_layer_call_fn_745103l��8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
E__inference_conv2d_17_layer_call_and_return_conditional_losses_745133w��8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
*__inference_conv2d_17_layer_call_fn_745122l��8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
E__inference_conv2d_18_layer_call_and_return_conditional_losses_745152w��8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
*__inference_conv2d_18_layer_call_fn_745142l��8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
E__inference_conv2d_19_layer_call_and_return_conditional_losses_745172w��8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
*__inference_conv2d_19_layer_call_fn_745161l��8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
D__inference_conv2d_1_layer_call_and_return_conditional_losses_744181x?@9�6
/�,
*�'
inputs�����������@
� "7�4
-�*
tensor_0������������
� �
)__inference_conv2d_1_layer_call_fn_744171m?@9�6
/�,
*�'
inputs�����������@
� ",�)
unknown�������������
E__inference_conv2d_20_layer_call_and_return_conditional_losses_745191w��8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
*__inference_conv2d_20_layer_call_fn_745181l��8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
E__inference_conv2d_21_layer_call_and_return_conditional_losses_744840y��9�6
/�,
*�'
inputs�����������@
� "6�3
,�)
tensor_0�����������
� �
*__inference_conv2d_21_layer_call_fn_744829n��9�6
/�,
*�'
inputs�����������@
� "+�(
unknown������������
D__inference_conv2d_2_layer_call_and_return_conditional_losses_744262wVW:�7
0�-
+�(
inputs������������
� "5�2
+�(
tensor_0���������@@�
� �
)__inference_conv2d_2_layer_call_fn_744252lVW:�7
0�-
+�(
inputs������������
� "*�'
unknown���������@@��
D__inference_conv2d_3_layer_call_and_return_conditional_losses_744860w��8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
)__inference_conv2d_3_layer_call_fn_744849l��8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
D__inference_conv2d_4_layer_call_and_return_conditional_losses_744879w��8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
)__inference_conv2d_4_layer_call_fn_744869l��8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
D__inference_conv2d_5_layer_call_and_return_conditional_losses_744899w��8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
)__inference_conv2d_5_layer_call_fn_744888l��8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
D__inference_conv2d_6_layer_call_and_return_conditional_losses_744918w��8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
)__inference_conv2d_6_layer_call_fn_744908l��8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
D__inference_conv2d_7_layer_call_and_return_conditional_losses_744938w��8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
)__inference_conv2d_7_layer_call_fn_744927l��8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
D__inference_conv2d_8_layer_call_and_return_conditional_losses_744957w��8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
)__inference_conv2d_8_layer_call_fn_744947l��8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
D__inference_conv2d_9_layer_call_and_return_conditional_losses_744977w��8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
)__inference_conv2d_9_layer_call_fn_744966l��8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
B__inference_conv2d_layer_call_and_return_conditional_losses_744100w()9�6
/�,
*�'
inputs�����������
� "6�3
,�)
tensor_0�����������@
� �
'__inference_conv2d_layer_call_fn_744090l()9�6
/�,
*�'
inputs�����������
� "+�(
unknown�����������@�
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_744758���J�G
@�=
;�8
inputs,����������������������������
� "F�C
<�9
tensor_0+���������������������������@
� �
3__inference_conv2d_transpose_1_layer_call_fn_744725���J�G
@�=
;�8
inputs,����������������������������
� ";�8
unknown+���������������������������@�
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_744654���J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
1__inference_conv2d_transpose_layer_call_fn_744621���J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
T__inference_instance_normalization_1_layer_call_and_return_conditional_losses_744233yHI:�7
0�-
+�(
inputs������������
� "7�4
-�*
tensor_0������������
� �
9__inference_instance_normalization_1_layer_call_fn_744190nHI:�7
0�-
+�(
inputs������������
� ",�)
unknown�������������
T__inference_instance_normalization_2_layer_call_and_return_conditional_losses_744314u_`8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
9__inference_instance_normalization_2_layer_call_fn_744271j_`8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
T__inference_instance_normalization_3_layer_call_and_return_conditional_losses_744706{��:�7
0�-
+�(
inputs������������
� "7�4
-�*
tensor_0������������
� �
9__inference_instance_normalization_3_layer_call_fn_744663p��:�7
0�-
+�(
inputs������������
� ",�)
unknown�������������
T__inference_instance_normalization_4_layer_call_and_return_conditional_losses_744810y��9�6
/�,
*�'
inputs�����������@
� "6�3
,�)
tensor_0�����������@
� �
9__inference_instance_normalization_4_layer_call_fn_744767n��9�6
/�,
*�'
inputs�����������@
� "+�(
unknown�����������@�
R__inference_instance_normalization_layer_call_and_return_conditional_losses_744152w129�6
/�,
*�'
inputs�����������@
� "6�3
,�)
tensor_0�����������@
� �
7__inference_instance_normalization_layer_call_fn_744109l129�6
/�,
*�'
inputs�����������@
� "+�(
unknown�����������@�
C__inference_re_lu_1_layer_call_and_return_conditional_losses_744243u:�7
0�-
+�(
inputs������������
� "7�4
-�*
tensor_0������������
� �
(__inference_re_lu_1_layer_call_fn_744238j:�7
0�-
+�(
inputs������������
� ",�)
unknown�������������
C__inference_re_lu_2_layer_call_and_return_conditional_losses_744324q8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
(__inference_re_lu_2_layer_call_fn_744319f8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
C__inference_re_lu_3_layer_call_and_return_conditional_losses_744716u:�7
0�-
+�(
inputs������������
� "7�4
-�*
tensor_0������������
� �
(__inference_re_lu_3_layer_call_fn_744711j:�7
0�-
+�(
inputs������������
� ",�)
unknown�������������
C__inference_re_lu_4_layer_call_and_return_conditional_losses_744820s9�6
/�,
*�'
inputs�����������@
� "6�3
,�)
tensor_0�����������@
� �
(__inference_re_lu_4_layer_call_fn_744815h9�6
/�,
*�'
inputs�����������@
� "+�(
unknown�����������@�
A__inference_re_lu_layer_call_and_return_conditional_losses_744162s9�6
/�,
*�'
inputs�����������@
� "6�3
,�)
tensor_0�����������@
� �
&__inference_re_lu_layer_call_fn_744157h9�6
/�,
*�'
inputs�����������@
� "+�(
unknown�����������@�
F__inference_residual_1_layer_call_and_return_conditional_losses_744388{����8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
+__inference_residual_1_layer_call_fn_744369p����8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
F__inference_residual_2_layer_call_and_return_conditional_losses_744420{����8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
+__inference_residual_2_layer_call_fn_744401p����8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
F__inference_residual_3_layer_call_and_return_conditional_losses_744452{����8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
+__inference_residual_3_layer_call_fn_744433p����8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
F__inference_residual_4_layer_call_and_return_conditional_losses_744484{����8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
+__inference_residual_4_layer_call_fn_744465p����8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
F__inference_residual_5_layer_call_and_return_conditional_losses_744516{����8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
+__inference_residual_5_layer_call_fn_744497p����8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
F__inference_residual_6_layer_call_and_return_conditional_losses_744548{����8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
+__inference_residual_6_layer_call_fn_744529p����8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
F__inference_residual_7_layer_call_and_return_conditional_losses_744580{����8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
+__inference_residual_7_layer_call_fn_744561p����8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
F__inference_residual_8_layer_call_and_return_conditional_losses_744612{����8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
+__inference_residual_8_layer_call_fn_744593p����8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
D__inference_residual_layer_call_and_return_conditional_losses_744356{����8�5
.�+
)�&
inputs���������@@�
� "5�2
+�(
tensor_0���������@@�
� �
)__inference_residual_layer_call_fn_744337p����8�5
.�+
)�&
inputs���������@@�
� "*�'
unknown���������@@��
F__inference_sequential_layer_call_and_return_conditional_losses_741872�h()12?@HIVW_`����������������������������������������������B�?
8�5
+�(
input_1�����������
p

 
� "6�3
,�)
tensor_0�����������
� �
F__inference_sequential_layer_call_and_return_conditional_losses_742017�h()12?@HIVW_`����������������������������������������������B�?
8�5
+�(
input_1�����������
p 

 
� "6�3
,�)
tensor_0�����������
� �
F__inference_sequential_layer_call_and_return_conditional_losses_743679�h()12?@HIVW_`����������������������������������������������A�>
7�4
*�'
inputs�����������
p

 
� "6�3
,�)
tensor_0�����������
� �
F__inference_sequential_layer_call_and_return_conditional_losses_744081�h()12?@HIVW_`����������������������������������������������A�>
7�4
*�'
inputs�����������
p 

 
� "6�3
,�)
tensor_0�����������
� �
+__inference_sequential_layer_call_fn_742284�h()12?@HIVW_`����������������������������������������������B�?
8�5
+�(
input_1�����������
p

 
� "+�(
unknown������������
+__inference_sequential_layer_call_fn_742550�h()12?@HIVW_`����������������������������������������������B�?
8�5
+�(
input_1�����������
p 

 
� "+�(
unknown������������
+__inference_sequential_layer_call_fn_743156�h()12?@HIVW_`����������������������������������������������A�>
7�4
*�'
inputs�����������
p

 
� "+�(
unknown������������
+__inference_sequential_layer_call_fn_743277�h()12?@HIVW_`����������������������������������������������A�>
7�4
*�'
inputs�����������
p 

 
� "+�(
unknown������������
$__inference_signature_wrapper_743035�h()12?@HIVW_`����������������������������������������������E�B
� 
;�8
6
input_1+�(
input_1�����������"?�<
:
	conv2d_21-�*
	conv2d_21�����������