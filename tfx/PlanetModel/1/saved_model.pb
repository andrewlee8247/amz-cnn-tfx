??
?/?/
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	??
E
AssignAddVariableOp
resource
value"dtype"
dtypetype?
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
.
Log1p
x"T
y"T"
Ttype:

2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
?
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	?
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
;
Minimum
x"T
y"T
z"T"
Ttype:

2	?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	
?
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
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
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?
9
VarIsInitializedOp
resource
is_initialized
?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.9.02v1.9.0-0-g25c197e023??
\
keras_learning_phase/inputConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
|
keras_learning_phasePlaceholderWithDefaultkeras_learning_phase/input*
dtype0
*
_output_shapes
: *
shape: 
?
input_layerPlaceholder*1
_output_shapes
:???????????*&
shape:???????????*
dtype0
?
.conv2d/kernel/Initializer/random_uniform/shapeConst*%
valueB"             * 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
:
?
,conv2d/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *OS?* 
_class
loc:@conv2d/kernel*
dtype0
?
,conv2d/kernel/Initializer/random_uniform/maxConst*
valueB
 *OS>* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
?
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: *

seed *
T0* 
_class
loc:@conv2d/kernel*
seed2 
?
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
?
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
?
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
?
conv2d/kernelVarHandleOp*
shape: *
dtype0*
_output_shapes
: *
shared_nameconv2d/kernel* 
_class
loc:@conv2d/kernel*
	container 
k
.conv2d/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d/kernel*
_output_shapes
: 
?
conv2d/kernel/AssignAssignVariableOpconv2d/kernel(conv2d/kernel/Initializer/random_uniform* 
_class
loc:@conv2d/kernel*
dtype0
?
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel* 
_class
loc:@conv2d/kernel*
dtype0*&
_output_shapes
: 
?
conv2d/bias/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB *    *
_class
loc:@conv2d/bias
?
conv2d/biasVarHandleOp*
shape: *
dtype0*
_output_shapes
: *
shared_nameconv2d/bias*
_class
loc:@conv2d/bias*
	container 
g
,conv2d/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d/bias*
_output_shapes
: 

conv2d/bias/AssignAssignVariableOpconv2d/biasconv2d/bias/Initializer/zeros*
_class
loc:@conv2d/bias*
dtype0
?
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes
: *
_class
loc:@conv2d/bias
e
conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
r
conv2d/Conv2D/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*&
_output_shapes
: 
?
conv2d/Conv2DConv2Dinput_layerconv2d/Conv2D/ReadVariableOp*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:??????????? *
	dilations
*
T0
e
conv2d/BiasAdd/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes
: 
?
conv2d/BiasAddBiasAddconv2d/Conv2Dconv2d/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*1
_output_shapes
:??????????? 
_
conv2d/ReluReluconv2d/BiasAdd*1
_output_shapes
:??????????? *
T0
?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:?????????@@ 
?
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"              *"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
:
?
.conv2d_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *?ѽ*"
_class
loc:@conv2d_1/kernel
?
.conv2d_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *??=*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
?
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv2d_1/kernel*
seed2 *
dtype0*&
_output_shapes
:  *

seed 
?
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_1/kernel
?
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  
?
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*&
_output_shapes
:  *
T0*"
_class
loc:@conv2d_1/kernel
?
conv2d_1/kernelVarHandleOp* 
shared_nameconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
	container *
shape:  *
dtype0*
_output_shapes
: 
o
0conv2d_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/kernel*
_output_shapes
: 
?
conv2d_1/kernel/AssignAssignVariableOpconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_1/kernel*
dtype0
?
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*&
_output_shapes
:  
?
conv2d_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB *    * 
_class
loc:@conv2d_1/bias
?
conv2d_1/biasVarHandleOp*
shared_nameconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
k
.conv2d_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/bias*
_output_shapes
: 
?
conv2d_1/bias/AssignAssignVariableOpconv2d_1/biasconv2d_1/bias/Initializer/zeros*
dtype0* 
_class
loc:@conv2d_1/bias
?
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 
g
conv2d_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
v
conv2d_1/Conv2D/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:  
?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPoolconv2d_1/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:?????????@@ *
	dilations

i
conv2d_1/BiasAdd/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
: 
?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2Dconv2d_1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:?????????@@ 
a
conv2d_1/ReluReluconv2d_1/BiasAdd*
T0*/
_output_shapes
:?????????@@ 
?
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:?????????   
d
flatten/ShapeShapemax_pooling2d_1/MaxPool*
T0*
out_type0*
_output_shapes
:
e
flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
flatten/strided_sliceStridedSliceflatten/Shapeflatten/strided_slice/stackflatten/strided_slice/stack_1flatten/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
b
flatten/Reshape/shape/1Const*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
flatten/Reshape/shapePackflatten/strided_sliceflatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
?
flatten/ReshapeReshapemax_pooling2d_1/MaxPoolflatten/Reshape/shape*
T0*
Tshape0*)
_output_shapes
:???????????
?
-dense/kernel/Initializer/random_uniform/shapeConst*
valueB" ?  ?   *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:
?
+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *PE]?*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
?
+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *PE]<*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
?
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*!
_output_shapes
:???*

seed *
T0*
_class
loc:@dense/kernel
?
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@dense/kernel
?
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*!
_output_shapes
:???*
T0*
_class
loc:@dense/kernel
?
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*!
_output_shapes
:???*
T0*
_class
loc:@dense/kernel
?
dense/kernelVarHandleOp*
	container *
shape:???*
dtype0*
_output_shapes
: *
shared_namedense/kernel*
_class
loc:@dense/kernel
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
?
dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
dtype0*
_class
loc:@dense/kernel
?
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_class
loc:@dense/kernel*
dtype0*!
_output_shapes
:???
?
dense/bias/Initializer/zerosConst*
valueB?*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes	
:?
?

dense/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_name
dense/bias*
_class
loc:@dense/bias*
	container *
shape:?
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
{
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
_class
loc:@dense/bias*
dtype0
?
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes	
:?*
_class
loc:@dense/bias
k
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0*!
_output_shapes
:???
?
dense/MatMulMatMulflatten/Reshapedense/MatMul/ReadVariableOp*
T0*
transpose_a( *(
_output_shapes
:??????????*
transpose_b( 
d
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes	
:?
?
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
data_formatNHWC*(
_output_shapes
:??????????*
T0
T

dense/ReluReludense/BiasAdd*(
_output_shapes
:??????????*
T0
[
dropout/IdentityIdentity
dense/Relu*
T0*(
_output_shapes
:??????????
?
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"?   ?   *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:
?
-dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *q??*!
_class
loc:@dense_1/kernel
?
-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *q?>*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
?
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_1/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed 
?
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
?
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel* 
_output_shapes
:
??
?
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel* 
_output_shapes
:
??
?
dense_1/kernelVarHandleOp*
shared_namedense_1/kernel*!
_class
loc:@dense_1/kernel*
	container *
shape:
??*
dtype0*
_output_shapes
: 
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
?
dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*!
_class
loc:@dense_1/kernel*
dtype0
?
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0* 
_output_shapes
:
??
?
dense_1/bias/Initializer/zerosConst*
valueB?*    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes	
:?
?
dense_1/biasVarHandleOp*
shared_namedense_1/bias*
_class
loc:@dense_1/bias*
	container *
shape:?*
dtype0*
_output_shapes
: 
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
?
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
_class
loc:@dense_1/bias*
dtype0
?
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes	
:?
n
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0* 
_output_shapes
:
??
?
dense_1/MatMulMatMuldropout/Identitydense_1/MatMul/ReadVariableOp*
transpose_a( *(
_output_shapes
:??????????*
transpose_b( *
T0
h
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes	
:?
?
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
data_formatNHWC*(
_output_shapes
:??????????*
T0
X
dense_1/ReluReludense_1/BiasAdd*
T0*(
_output_shapes
:??????????
_
dropout_1/IdentityIdentitydense_1/Relu*(
_output_shapes
:??????????*
T0
?
/weather/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"?      *!
_class
loc:@weather/kernel
?
-weather/kernel/Initializer/random_uniform/minConst*
valueB
 *JQZ?*!
_class
loc:@weather/kernel*
dtype0*
_output_shapes
: 
?
-weather/kernel/Initializer/random_uniform/maxConst*
valueB
 *JQZ>*!
_class
loc:@weather/kernel*
dtype0*
_output_shapes
: 
?
7weather/kernel/Initializer/random_uniform/RandomUniformRandomUniform/weather/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	?*

seed *
T0*!
_class
loc:@weather/kernel*
seed2 
?
-weather/kernel/Initializer/random_uniform/subSub-weather/kernel/Initializer/random_uniform/max-weather/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@weather/kernel*
_output_shapes
: 
?
-weather/kernel/Initializer/random_uniform/mulMul7weather/kernel/Initializer/random_uniform/RandomUniform-weather/kernel/Initializer/random_uniform/sub*
_output_shapes
:	?*
T0*!
_class
loc:@weather/kernel
?
)weather/kernel/Initializer/random_uniformAdd-weather/kernel/Initializer/random_uniform/mul-weather/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@weather/kernel*
_output_shapes
:	?
?
weather/kernelVarHandleOp*
shape:	?*
dtype0*
_output_shapes
: *
shared_nameweather/kernel*!
_class
loc:@weather/kernel*
	container 
m
/weather/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpweather/kernel*
_output_shapes
: 
?
weather/kernel/AssignAssignVariableOpweather/kernel)weather/kernel/Initializer/random_uniform*!
_class
loc:@weather/kernel*
dtype0
?
"weather/kernel/Read/ReadVariableOpReadVariableOpweather/kernel*!
_class
loc:@weather/kernel*
dtype0*
_output_shapes
:	?
?
weather/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@weather/bias*
dtype0*
_output_shapes
:
?
weather/biasVarHandleOp*
shared_nameweather/bias*
_class
loc:@weather/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
i
-weather/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpweather/bias*
_output_shapes
: 
?
weather/bias/AssignAssignVariableOpweather/biasweather/bias/Initializer/zeros*
_class
loc:@weather/bias*
dtype0
?
 weather/bias/Read/ReadVariableOpReadVariableOpweather/bias*
_class
loc:@weather/bias*
dtype0*
_output_shapes
:
m
weather/MatMul/ReadVariableOpReadVariableOpweather/kernel*
dtype0*
_output_shapes
:	?
?
weather/MatMulMatMuldropout_1/Identityweather/MatMul/ReadVariableOp*
transpose_a( *'
_output_shapes
:?????????*
transpose_b( *
T0
g
weather/BiasAdd/ReadVariableOpReadVariableOpweather/bias*
dtype0*
_output_shapes
:
?
weather/BiasAddBiasAddweather/MatMulweather/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:?????????
]
weather/SoftmaxSoftmaxweather/BiasAdd*
T0*'
_output_shapes
:?????????
?
.ground/kernel/Initializer/random_uniform/shapeConst*
valueB"?      * 
_class
loc:@ground/kernel*
dtype0*
_output_shapes
:
?
,ground/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *.<S?* 
_class
loc:@ground/kernel
?
,ground/kernel/Initializer/random_uniform/maxConst*
valueB
 *.<S>* 
_class
loc:@ground/kernel*
dtype0*
_output_shapes
: 
?
6ground/kernel/Initializer/random_uniform/RandomUniformRandomUniform.ground/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	?*

seed *
T0* 
_class
loc:@ground/kernel*
seed2 
?
,ground/kernel/Initializer/random_uniform/subSub,ground/kernel/Initializer/random_uniform/max,ground/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0* 
_class
loc:@ground/kernel
?
,ground/kernel/Initializer/random_uniform/mulMul6ground/kernel/Initializer/random_uniform/RandomUniform,ground/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@ground/kernel*
_output_shapes
:	?
?
(ground/kernel/Initializer/random_uniformAdd,ground/kernel/Initializer/random_uniform/mul,ground/kernel/Initializer/random_uniform/min*
_output_shapes
:	?*
T0* 
_class
loc:@ground/kernel
?
ground/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameground/kernel* 
_class
loc:@ground/kernel*
	container *
shape:	?
k
.ground/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpground/kernel*
_output_shapes
: 
?
ground/kernel/AssignAssignVariableOpground/kernel(ground/kernel/Initializer/random_uniform* 
_class
loc:@ground/kernel*
dtype0
?
!ground/kernel/Read/ReadVariableOpReadVariableOpground/kernel*
dtype0*
_output_shapes
:	?* 
_class
loc:@ground/kernel
?
ground/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@ground/bias*
dtype0*
_output_shapes
:
?
ground/biasVarHandleOp*
shared_nameground/bias*
_class
loc:@ground/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
g
,ground/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpground/bias*
_output_shapes
: 

ground/bias/AssignAssignVariableOpground/biasground/bias/Initializer/zeros*
_class
loc:@ground/bias*
dtype0
?
ground/bias/Read/ReadVariableOpReadVariableOpground/bias*
_class
loc:@ground/bias*
dtype0*
_output_shapes
:
k
ground/MatMul/ReadVariableOpReadVariableOpground/kernel*
dtype0*
_output_shapes
:	?
?
ground/MatMulMatMuldropout_1/Identityground/MatMul/ReadVariableOp*
T0*
transpose_a( *'
_output_shapes
:?????????*
transpose_b( 
e
ground/BiasAdd/ReadVariableOpReadVariableOpground/bias*
dtype0*
_output_shapes
:
?
ground/BiasAddBiasAddground/MatMulground/BiasAdd/ReadVariableOp*
data_formatNHWC*'
_output_shapes
:?????????*
T0
[
ground/SigmoidSigmoidground/BiasAdd*'
_output_shapes
:?????????*
T0
l
PlaceholderPlaceholder*
dtype0*&
_output_shapes
: *
shape: 
M
AssignVariableOpAssignVariableOpconv2d/kernelPlaceholder*
dtype0
w
ReadVariableOpReadVariableOpconv2d/kernel^AssignVariableOp*
dtype0*&
_output_shapes
: 
V
Placeholder_1Placeholder*
dtype0*
_output_shapes
: *
shape: 
O
AssignVariableOp_1AssignVariableOpconv2d/biasPlaceholder_1*
dtype0
m
ReadVariableOp_1ReadVariableOpconv2d/bias^AssignVariableOp_1*
dtype0*
_output_shapes
: 
n
Placeholder_2Placeholder*
dtype0*&
_output_shapes
:  *
shape:  
S
AssignVariableOp_2AssignVariableOpconv2d_1/kernelPlaceholder_2*
dtype0
}
ReadVariableOp_2ReadVariableOpconv2d_1/kernel^AssignVariableOp_2*
dtype0*&
_output_shapes
:  
V
Placeholder_3Placeholder*
dtype0*
_output_shapes
: *
shape: 
Q
AssignVariableOp_3AssignVariableOpconv2d_1/biasPlaceholder_3*
dtype0
o
ReadVariableOp_3ReadVariableOpconv2d_1/bias^AssignVariableOp_3*
dtype0*
_output_shapes
: 
d
Placeholder_4Placeholder*
dtype0*!
_output_shapes
:???*
shape:???
P
AssignVariableOp_4AssignVariableOpdense/kernelPlaceholder_4*
dtype0
u
ReadVariableOp_4ReadVariableOpdense/kernel^AssignVariableOp_4*
dtype0*!
_output_shapes
:???
X
Placeholder_5Placeholder*
dtype0*
_output_shapes	
:?*
shape:?
N
AssignVariableOp_5AssignVariableOp
dense/biasPlaceholder_5*
dtype0
m
ReadVariableOp_5ReadVariableOp
dense/bias^AssignVariableOp_5*
dtype0*
_output_shapes	
:?
b
Placeholder_6Placeholder*
dtype0* 
_output_shapes
:
??*
shape:
??
R
AssignVariableOp_6AssignVariableOpdense_1/kernelPlaceholder_6*
dtype0
v
ReadVariableOp_6ReadVariableOpdense_1/kernel^AssignVariableOp_6*
dtype0* 
_output_shapes
:
??
X
Placeholder_7Placeholder*
dtype0*
_output_shapes	
:?*
shape:?
P
AssignVariableOp_7AssignVariableOpdense_1/biasPlaceholder_7*
dtype0
o
ReadVariableOp_7ReadVariableOpdense_1/bias^AssignVariableOp_7*
dtype0*
_output_shapes	
:?
`
Placeholder_8Placeholder*
dtype0*
_output_shapes
:	?*
shape:	?
R
AssignVariableOp_8AssignVariableOpweather/kernelPlaceholder_8*
dtype0
u
ReadVariableOp_8ReadVariableOpweather/kernel^AssignVariableOp_8*
dtype0*
_output_shapes
:	?
V
Placeholder_9Placeholder*
dtype0*
_output_shapes
:*
shape:
P
AssignVariableOp_9AssignVariableOpweather/biasPlaceholder_9*
dtype0
n
ReadVariableOp_9ReadVariableOpweather/bias^AssignVariableOp_9*
dtype0*
_output_shapes
:
a
Placeholder_10Placeholder*
dtype0*
_output_shapes
:	?*
shape:	?
S
AssignVariableOp_10AssignVariableOpground/kernelPlaceholder_10*
dtype0
v
ReadVariableOp_10ReadVariableOpground/kernel^AssignVariableOp_10*
dtype0*
_output_shapes
:	?
W
Placeholder_11Placeholder*
dtype0*
_output_shapes
:*
shape:
Q
AssignVariableOp_11AssignVariableOpground/biasPlaceholder_11*
dtype0
o
ReadVariableOp_11ReadVariableOpground/bias^AssignVariableOp_11*
dtype0*
_output_shapes
:
O
VarIsInitializedOpVarIsInitializedOpconv2d/kernel*
_output_shapes
: 
O
VarIsInitializedOp_1VarIsInitializedOpconv2d/bias*
_output_shapes
: 
S
VarIsInitializedOp_2VarIsInitializedOpconv2d_1/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_3VarIsInitializedOpconv2d_1/bias*
_output_shapes
: 
P
VarIsInitializedOp_4VarIsInitializedOpdense/kernel*
_output_shapes
: 
N
VarIsInitializedOp_5VarIsInitializedOp
dense/bias*
_output_shapes
: 
R
VarIsInitializedOp_6VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
P
VarIsInitializedOp_7VarIsInitializedOpdense_1/bias*
_output_shapes
: 
R
VarIsInitializedOp_8VarIsInitializedOpweather/kernel*
_output_shapes
: 
P
VarIsInitializedOp_9VarIsInitializedOpweather/bias*
_output_shapes
: 
R
VarIsInitializedOp_10VarIsInitializedOpground/kernel*
_output_shapes
: 
P
VarIsInitializedOp_11VarIsInitializedOpground/bias*
_output_shapes
: 
?
initNoOp^conv2d/bias/Assign^conv2d/kernel/Assign^conv2d_1/bias/Assign^conv2d_1/kernel/Assign^dense/bias/Assign^dense/kernel/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^ground/bias/Assign^ground/kernel/Assign^weather/bias/Assign^weather/kernel/Assign
k
)Adam/iterations/Initializer/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
Adam/iterationsVarHandleOp*
shape: * 
shared_nameAdam/iterations*
dtype0	*
	container *
_output_shapes
: 
o
0Adam/iterations/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/iterations*
_output_shapes
: 
?
Adam/iterations/AssignAssignVariableOpAdam/iterations)Adam/iterations/Initializer/initial_value*
dtype0	*"
_class
loc:@Adam/iterations
?
#Adam/iterations/Read/ReadVariableOpReadVariableOpAdam/iterations*"
_class
loc:@Adam/iterations*
dtype0	*
_output_shapes
: 
f
!Adam/lr/Initializer/initial_valueConst*
valueB
 *o?:*
dtype0*
_output_shapes
: 
s
Adam/lrVarHandleOp*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name	Adam/lr
_
(Adam/lr/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/lr*
_output_shapes
: 
w
Adam/lr/AssignAssignVariableOpAdam/lr!Adam/lr/Initializer/initial_value*
_class
loc:@Adam/lr*
dtype0
w
Adam/lr/Read/ReadVariableOpReadVariableOpAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 
j
%Adam/beta_1/Initializer/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
{
Adam/beta_1VarHandleOp*
shape: *
shared_nameAdam/beta_1*
dtype0*
	container *
_output_shapes
: 
g
,Adam/beta_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/beta_1*
_output_shapes
: 
?
Adam/beta_1/AssignAssignVariableOpAdam/beta_1%Adam/beta_1/Initializer/initial_value*
_class
loc:@Adam/beta_1*
dtype0
?
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 
j
%Adam/beta_2/Initializer/initial_valueConst*
valueB
 *w??*
dtype0*
_output_shapes
: 
{
Adam/beta_2VarHandleOp*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_nameAdam/beta_2
g
,Adam/beta_2/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/beta_2*
_output_shapes
: 
?
Adam/beta_2/AssignAssignVariableOpAdam/beta_2%Adam/beta_2/Initializer/initial_value*
dtype0*
_class
loc:@Adam/beta_2
?
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 
i
$Adam/decay/Initializer/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
y

Adam/decayVarHandleOp*
shared_name
Adam/decay*
dtype0*
	container *
_output_shapes
: *
shape: 
e
+Adam/decay/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Adam/decay*
_output_shapes
: 
?
Adam/decay/AssignAssignVariableOp
Adam/decay$Adam/decay/Initializer/initial_value*
dtype0*
_class
loc:@Adam/decay
?
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 
?
weather_targetPlaceholder*
dtype0*0
_output_shapes
:??????????????????*%
shape:??????????????????
?
ground_targetPlaceholder*
dtype0*0
_output_shapes
:??????????????????*%
shape:??????????????????
i
weather_sample_weights/inputConst*
valueB*  ??*
dtype0*
_output_shapes
:
?
weather_sample_weightsPlaceholderWithDefaultweather_sample_weights/input*
dtype0*#
_output_shapes
:?????????*
shape:?????????
h
ground_sample_weights/inputConst*
valueB*  ??*
dtype0*
_output_shapes
:
?
ground_sample_weightsPlaceholderWithDefaultground_sample_weights/input*
dtype0*#
_output_shapes
:?????????*
shape:?????????
i
'loss/weather_loss/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
?
loss/weather_loss/SumSumweather/Softmax'loss/weather_loss/Sum/reduction_indices*'
_output_shapes
:?????????*

Tidx0*
	keep_dims(*
T0
~
loss/weather_loss/truedivRealDivweather/Softmaxloss/weather_loss/Sum*
T0*'
_output_shapes
:?????????
\
loss/weather_loss/ConstConst*
valueB
 *???3*
dtype0*
_output_shapes
: 
\
loss/weather_loss/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
o
loss/weather_loss/subSubloss/weather_loss/sub/xloss/weather_loss/Const*
T0*
_output_shapes
: 
?
'loss/weather_loss/clip_by_value/MinimumMinimumloss/weather_loss/truedivloss/weather_loss/sub*
T0*'
_output_shapes
:?????????
?
loss/weather_loss/clip_by_valueMaximum'loss/weather_loss/clip_by_value/Minimumloss/weather_loss/Const*'
_output_shapes
:?????????*
T0
o
loss/weather_loss/LogLogloss/weather_loss/clip_by_value*
T0*'
_output_shapes
:?????????
u
loss/weather_loss/mulMulweather_targetloss/weather_loss/Log*
T0*'
_output_shapes
:?????????
k
)loss/weather_loss/Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
?
loss/weather_loss/Sum_1Sumloss/weather_loss/mul)loss/weather_loss/Sum_1/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:?????????
c
loss/weather_loss/NegNegloss/weather_loss/Sum_1*#
_output_shapes
:?????????*
T0
k
(loss/weather_loss/Mean/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
?
loss/weather_loss/MeanMeanloss/weather_loss/Neg(loss/weather_loss/Mean/reduction_indices*
T0*#
_output_shapes
:?????????*

Tidx0*
	keep_dims( 
|
loss/weather_loss/mul_1Mulloss/weather_loss/Meanweather_sample_weights*#
_output_shapes
:?????????*
T0
a
loss/weather_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
loss/weather_loss/NotEqualNotEqualweather_sample_weightsloss/weather_loss/NotEqual/y*
T0*#
_output_shapes
:?????????
w
loss/weather_loss/CastCastloss/weather_loss/NotEqual*

DstT0*#
_output_shapes
:?????????*

SrcT0

c
loss/weather_loss/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
?
loss/weather_loss/Mean_1Meanloss/weather_loss/Castloss/weather_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
?
loss/weather_loss/truediv_1RealDivloss/weather_loss/mul_1loss/weather_loss/Mean_1*
T0*#
_output_shapes
:?????????
c
loss/weather_loss/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
?
loss/weather_loss/Mean_2Meanloss/weather_loss/truediv_1loss/weather_loss/Const_2*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
O

loss/mul/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
V
loss/mulMul
loss/mul/xloss/weather_loss/Mean_2*
_output_shapes
: *
T0
[
loss/ground_loss/ConstConst*
valueB
 *???3*
dtype0*
_output_shapes
: 
[
loss/ground_loss/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
l
loss/ground_loss/subSubloss/ground_loss/sub/xloss/ground_loss/Const*
T0*
_output_shapes
: 
?
&loss/ground_loss/clip_by_value/MinimumMinimumground/Sigmoidloss/ground_loss/sub*
T0*'
_output_shapes
:?????????
?
loss/ground_loss/clip_by_valueMaximum&loss/ground_loss/clip_by_value/Minimumloss/ground_loss/Const*'
_output_shapes
:?????????*
T0
]
loss/ground_loss/sub_1/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
loss/ground_loss/sub_1Subloss/ground_loss/sub_1/xloss/ground_loss/clip_by_value*
T0*'
_output_shapes
:?????????
?
loss/ground_loss/truedivRealDivloss/ground_loss/clip_by_valueloss/ground_loss/sub_1*'
_output_shapes
:?????????*
T0
g
loss/ground_loss/LogLogloss/ground_loss/truediv*'
_output_shapes
:?????????*
T0
~
)loss/ground_loss/logistic_loss/zeros_like	ZerosLikeloss/ground_loss/Log*
T0*'
_output_shapes
:?????????
?
+loss/ground_loss/logistic_loss/GreaterEqualGreaterEqualloss/ground_loss/Log)loss/ground_loss/logistic_loss/zeros_like*
T0*'
_output_shapes
:?????????
?
%loss/ground_loss/logistic_loss/SelectSelect+loss/ground_loss/logistic_loss/GreaterEqualloss/ground_loss/Log)loss/ground_loss/logistic_loss/zeros_like*'
_output_shapes
:?????????*
T0
q
"loss/ground_loss/logistic_loss/NegNegloss/ground_loss/Log*
T0*'
_output_shapes
:?????????
?
'loss/ground_loss/logistic_loss/Select_1Select+loss/ground_loss/logistic_loss/GreaterEqual"loss/ground_loss/logistic_loss/Negloss/ground_loss/Log*
T0*'
_output_shapes
:?????????
?
"loss/ground_loss/logistic_loss/mulMulloss/ground_loss/Logground_target*
T0*'
_output_shapes
:?????????
?
"loss/ground_loss/logistic_loss/subSub%loss/ground_loss/logistic_loss/Select"loss/ground_loss/logistic_loss/mul*
T0*'
_output_shapes
:?????????
?
"loss/ground_loss/logistic_loss/ExpExp'loss/ground_loss/logistic_loss/Select_1*'
_output_shapes
:?????????*
T0
?
$loss/ground_loss/logistic_loss/Log1pLog1p"loss/ground_loss/logistic_loss/Exp*'
_output_shapes
:?????????*
T0
?
loss/ground_loss/logistic_lossAdd"loss/ground_loss/logistic_loss/sub$loss/ground_loss/logistic_loss/Log1p*
T0*'
_output_shapes
:?????????
r
'loss/ground_loss/Mean/reduction_indicesConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
loss/ground_loss/MeanMeanloss/ground_loss/logistic_loss'loss/ground_loss/Mean/reduction_indices*
T0*#
_output_shapes
:?????????*

Tidx0*
	keep_dims( 
l
)loss/ground_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
?
loss/ground_loss/Mean_1Meanloss/ground_loss/Mean)loss/ground_loss/Mean_1/reduction_indices*#
_output_shapes
:?????????*

Tidx0*
	keep_dims( *
T0
y
loss/ground_loss/mulMulloss/ground_loss/Mean_1ground_sample_weights*#
_output_shapes
:?????????*
T0
`
loss/ground_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
loss/ground_loss/NotEqualNotEqualground_sample_weightsloss/ground_loss/NotEqual/y*
T0*#
_output_shapes
:?????????
u
loss/ground_loss/CastCastloss/ground_loss/NotEqual*

DstT0*#
_output_shapes
:?????????*

SrcT0

b
loss/ground_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
loss/ground_loss/Mean_2Meanloss/ground_loss/Castloss/ground_loss/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
?
loss/ground_loss/truediv_1RealDivloss/ground_loss/mulloss/ground_loss/Mean_2*
T0*#
_output_shapes
:?????????
b
loss/ground_loss/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
?
loss/ground_loss/Mean_3Meanloss/ground_loss/truediv_1loss/ground_loss/Const_2*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
Q
loss/mul_1/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
Y

loss/mul_1Mulloss/mul_1/xloss/ground_loss/Mean_3*
T0*
_output_shapes
: 
F
loss/addAddloss/mul
loss/mul_1*
T0*
_output_shapes
: 
}
training/Adam/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB *
_class
loc:@loss/add
?
!training/Adam/gradients/grad_ys_0Const*
valueB
 *  ??*
_class
loc:@loss/add*
dtype0*
_output_shapes
: 
?
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
T0*

index_type0*
_class
loc:@loss/add*
_output_shapes
: 
?
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/weather_loss/Mean_2*
_output_shapes
: *
T0*
_class
loc:@loss/mul
?
+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
?
+training/Adam/gradients/loss/mul_1_grad/MulMultraining/Adam/gradients/Fillloss/ground_loss/Mean_3*
T0*
_class
loc:@loss/mul_1*
_output_shapes
: 
?
-training/Adam/gradients/loss/mul_1_grad/Mul_1Multraining/Adam/gradients/Fillloss/mul_1/x*
T0*
_class
loc:@loss/mul_1*
_output_shapes
: 
?
Ctraining/Adam/gradients/loss/weather_loss/Mean_2_grad/Reshape/shapeConst*
valueB:*+
_class!
loc:@loss/weather_loss/Mean_2*
dtype0*
_output_shapes
:
?
=training/Adam/gradients/loss/weather_loss/Mean_2_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Ctraining/Adam/gradients/loss/weather_loss/Mean_2_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0*+
_class!
loc:@loss/weather_loss/Mean_2
?
;training/Adam/gradients/loss/weather_loss/Mean_2_grad/ShapeShapeloss/weather_loss/truediv_1*
T0*
out_type0*+
_class!
loc:@loss/weather_loss/Mean_2*
_output_shapes
:
?
:training/Adam/gradients/loss/weather_loss/Mean_2_grad/TileTile=training/Adam/gradients/loss/weather_loss/Mean_2_grad/Reshape;training/Adam/gradients/loss/weather_loss/Mean_2_grad/Shape*#
_output_shapes
:?????????*

Tmultiples0*
T0*+
_class!
loc:@loss/weather_loss/Mean_2
?
=training/Adam/gradients/loss/weather_loss/Mean_2_grad/Shape_1Shapeloss/weather_loss/truediv_1*
T0*
out_type0*+
_class!
loc:@loss/weather_loss/Mean_2*
_output_shapes
:
?
=training/Adam/gradients/loss/weather_loss/Mean_2_grad/Shape_2Const*
valueB *+
_class!
loc:@loss/weather_loss/Mean_2*
dtype0*
_output_shapes
: 
?
;training/Adam/gradients/loss/weather_loss/Mean_2_grad/ConstConst*
valueB: *+
_class!
loc:@loss/weather_loss/Mean_2*
dtype0*
_output_shapes
:
?
:training/Adam/gradients/loss/weather_loss/Mean_2_grad/ProdProd=training/Adam/gradients/loss/weather_loss/Mean_2_grad/Shape_1;training/Adam/gradients/loss/weather_loss/Mean_2_grad/Const*

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@loss/weather_loss/Mean_2*
_output_shapes
: 
?
=training/Adam/gradients/loss/weather_loss/Mean_2_grad/Const_1Const*
valueB: *+
_class!
loc:@loss/weather_loss/Mean_2*
dtype0*
_output_shapes
:
?
<training/Adam/gradients/loss/weather_loss/Mean_2_grad/Prod_1Prod=training/Adam/gradients/loss/weather_loss/Mean_2_grad/Shape_2=training/Adam/gradients/loss/weather_loss/Mean_2_grad/Const_1*
T0*+
_class!
loc:@loss/weather_loss/Mean_2*
_output_shapes
: *

Tidx0*
	keep_dims( 
?
?training/Adam/gradients/loss/weather_loss/Mean_2_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*+
_class!
loc:@loss/weather_loss/Mean_2
?
=training/Adam/gradients/loss/weather_loss/Mean_2_grad/MaximumMaximum<training/Adam/gradients/loss/weather_loss/Mean_2_grad/Prod_1?training/Adam/gradients/loss/weather_loss/Mean_2_grad/Maximum/y*
T0*+
_class!
loc:@loss/weather_loss/Mean_2*
_output_shapes
: 
?
>training/Adam/gradients/loss/weather_loss/Mean_2_grad/floordivFloorDiv:training/Adam/gradients/loss/weather_loss/Mean_2_grad/Prod=training/Adam/gradients/loss/weather_loss/Mean_2_grad/Maximum*
T0*+
_class!
loc:@loss/weather_loss/Mean_2*
_output_shapes
: 
?
:training/Adam/gradients/loss/weather_loss/Mean_2_grad/CastCast>training/Adam/gradients/loss/weather_loss/Mean_2_grad/floordiv*

SrcT0*+
_class!
loc:@loss/weather_loss/Mean_2*

DstT0*
_output_shapes
: 
?
=training/Adam/gradients/loss/weather_loss/Mean_2_grad/truedivRealDiv:training/Adam/gradients/loss/weather_loss/Mean_2_grad/Tile:training/Adam/gradients/loss/weather_loss/Mean_2_grad/Cast*#
_output_shapes
:?????????*
T0*+
_class!
loc:@loss/weather_loss/Mean_2
?
Btraining/Adam/gradients/loss/ground_loss/Mean_3_grad/Reshape/shapeConst*
valueB:**
_class 
loc:@loss/ground_loss/Mean_3*
dtype0*
_output_shapes
:
?
<training/Adam/gradients/loss/ground_loss/Mean_3_grad/ReshapeReshape-training/Adam/gradients/loss/mul_1_grad/Mul_1Btraining/Adam/gradients/loss/ground_loss/Mean_3_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0**
_class 
loc:@loss/ground_loss/Mean_3
?
:training/Adam/gradients/loss/ground_loss/Mean_3_grad/ShapeShapeloss/ground_loss/truediv_1*
T0*
out_type0**
_class 
loc:@loss/ground_loss/Mean_3*
_output_shapes
:
?
9training/Adam/gradients/loss/ground_loss/Mean_3_grad/TileTile<training/Adam/gradients/loss/ground_loss/Mean_3_grad/Reshape:training/Adam/gradients/loss/ground_loss/Mean_3_grad/Shape*#
_output_shapes
:?????????*

Tmultiples0*
T0**
_class 
loc:@loss/ground_loss/Mean_3
?
<training/Adam/gradients/loss/ground_loss/Mean_3_grad/Shape_1Shapeloss/ground_loss/truediv_1*
T0*
out_type0**
_class 
loc:@loss/ground_loss/Mean_3*
_output_shapes
:
?
<training/Adam/gradients/loss/ground_loss/Mean_3_grad/Shape_2Const*
valueB **
_class 
loc:@loss/ground_loss/Mean_3*
dtype0*
_output_shapes
: 
?
:training/Adam/gradients/loss/ground_loss/Mean_3_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: **
_class 
loc:@loss/ground_loss/Mean_3
?
9training/Adam/gradients/loss/ground_loss/Mean_3_grad/ProdProd<training/Adam/gradients/loss/ground_loss/Mean_3_grad/Shape_1:training/Adam/gradients/loss/ground_loss/Mean_3_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0**
_class 
loc:@loss/ground_loss/Mean_3
?
<training/Adam/gradients/loss/ground_loss/Mean_3_grad/Const_1Const*
valueB: **
_class 
loc:@loss/ground_loss/Mean_3*
dtype0*
_output_shapes
:
?
;training/Adam/gradients/loss/ground_loss/Mean_3_grad/Prod_1Prod<training/Adam/gradients/loss/ground_loss/Mean_3_grad/Shape_2<training/Adam/gradients/loss/ground_loss/Mean_3_grad/Const_1*

Tidx0*
	keep_dims( *
T0**
_class 
loc:@loss/ground_loss/Mean_3*
_output_shapes
: 
?
>training/Adam/gradients/loss/ground_loss/Mean_3_grad/Maximum/yConst*
value	B :**
_class 
loc:@loss/ground_loss/Mean_3*
dtype0*
_output_shapes
: 
?
<training/Adam/gradients/loss/ground_loss/Mean_3_grad/MaximumMaximum;training/Adam/gradients/loss/ground_loss/Mean_3_grad/Prod_1>training/Adam/gradients/loss/ground_loss/Mean_3_grad/Maximum/y*
T0**
_class 
loc:@loss/ground_loss/Mean_3*
_output_shapes
: 
?
=training/Adam/gradients/loss/ground_loss/Mean_3_grad/floordivFloorDiv9training/Adam/gradients/loss/ground_loss/Mean_3_grad/Prod<training/Adam/gradients/loss/ground_loss/Mean_3_grad/Maximum*
_output_shapes
: *
T0**
_class 
loc:@loss/ground_loss/Mean_3
?
9training/Adam/gradients/loss/ground_loss/Mean_3_grad/CastCast=training/Adam/gradients/loss/ground_loss/Mean_3_grad/floordiv*

SrcT0**
_class 
loc:@loss/ground_loss/Mean_3*

DstT0*
_output_shapes
: 
?
<training/Adam/gradients/loss/ground_loss/Mean_3_grad/truedivRealDiv9training/Adam/gradients/loss/ground_loss/Mean_3_grad/Tile9training/Adam/gradients/loss/ground_loss/Mean_3_grad/Cast*#
_output_shapes
:?????????*
T0**
_class 
loc:@loss/ground_loss/Mean_3
?
>training/Adam/gradients/loss/weather_loss/truediv_1_grad/ShapeShapeloss/weather_loss/mul_1*
T0*
out_type0*.
_class$
" loc:@loss/weather_loss/truediv_1*
_output_shapes
:
?
@training/Adam/gradients/loss/weather_loss/truediv_1_grad/Shape_1Const*
valueB *.
_class$
" loc:@loss/weather_loss/truediv_1*
dtype0*
_output_shapes
: 
?
Ntraining/Adam/gradients/loss/weather_loss/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgs>training/Adam/gradients/loss/weather_loss/truediv_1_grad/Shape@training/Adam/gradients/loss/weather_loss/truediv_1_grad/Shape_1*
T0*.
_class$
" loc:@loss/weather_loss/truediv_1*2
_output_shapes 
:?????????:?????????
?
@training/Adam/gradients/loss/weather_loss/truediv_1_grad/RealDivRealDiv=training/Adam/gradients/loss/weather_loss/Mean_2_grad/truedivloss/weather_loss/Mean_1*
T0*.
_class$
" loc:@loss/weather_loss/truediv_1*#
_output_shapes
:?????????
?
<training/Adam/gradients/loss/weather_loss/truediv_1_grad/SumSum@training/Adam/gradients/loss/weather_loss/truediv_1_grad/RealDivNtraining/Adam/gradients/loss/weather_loss/truediv_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*.
_class$
" loc:@loss/weather_loss/truediv_1
?
@training/Adam/gradients/loss/weather_loss/truediv_1_grad/ReshapeReshape<training/Adam/gradients/loss/weather_loss/truediv_1_grad/Sum>training/Adam/gradients/loss/weather_loss/truediv_1_grad/Shape*
T0*
Tshape0*.
_class$
" loc:@loss/weather_loss/truediv_1*#
_output_shapes
:?????????
?
<training/Adam/gradients/loss/weather_loss/truediv_1_grad/NegNegloss/weather_loss/mul_1*
T0*.
_class$
" loc:@loss/weather_loss/truediv_1*#
_output_shapes
:?????????
?
Btraining/Adam/gradients/loss/weather_loss/truediv_1_grad/RealDiv_1RealDiv<training/Adam/gradients/loss/weather_loss/truediv_1_grad/Negloss/weather_loss/Mean_1*
T0*.
_class$
" loc:@loss/weather_loss/truediv_1*#
_output_shapes
:?????????
?
Btraining/Adam/gradients/loss/weather_loss/truediv_1_grad/RealDiv_2RealDivBtraining/Adam/gradients/loss/weather_loss/truediv_1_grad/RealDiv_1loss/weather_loss/Mean_1*#
_output_shapes
:?????????*
T0*.
_class$
" loc:@loss/weather_loss/truediv_1
?
<training/Adam/gradients/loss/weather_loss/truediv_1_grad/mulMul=training/Adam/gradients/loss/weather_loss/Mean_2_grad/truedivBtraining/Adam/gradients/loss/weather_loss/truediv_1_grad/RealDiv_2*#
_output_shapes
:?????????*
T0*.
_class$
" loc:@loss/weather_loss/truediv_1
?
>training/Adam/gradients/loss/weather_loss/truediv_1_grad/Sum_1Sum<training/Adam/gradients/loss/weather_loss/truediv_1_grad/mulPtraining/Adam/gradients/loss/weather_loss/truediv_1_grad/BroadcastGradientArgs:1*
T0*.
_class$
" loc:@loss/weather_loss/truediv_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
Btraining/Adam/gradients/loss/weather_loss/truediv_1_grad/Reshape_1Reshape>training/Adam/gradients/loss/weather_loss/truediv_1_grad/Sum_1@training/Adam/gradients/loss/weather_loss/truediv_1_grad/Shape_1*
T0*
Tshape0*.
_class$
" loc:@loss/weather_loss/truediv_1*
_output_shapes
: 
?
=training/Adam/gradients/loss/ground_loss/truediv_1_grad/ShapeShapeloss/ground_loss/mul*
_output_shapes
:*
T0*
out_type0*-
_class#
!loc:@loss/ground_loss/truediv_1
?
?training/Adam/gradients/loss/ground_loss/truediv_1_grad/Shape_1Const*
valueB *-
_class#
!loc:@loss/ground_loss/truediv_1*
dtype0*
_output_shapes
: 
?
Mtraining/Adam/gradients/loss/ground_loss/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/loss/ground_loss/truediv_1_grad/Shape?training/Adam/gradients/loss/ground_loss/truediv_1_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0*-
_class#
!loc:@loss/ground_loss/truediv_1
?
?training/Adam/gradients/loss/ground_loss/truediv_1_grad/RealDivRealDiv<training/Adam/gradients/loss/ground_loss/Mean_3_grad/truedivloss/ground_loss/Mean_2*
T0*-
_class#
!loc:@loss/ground_loss/truediv_1*#
_output_shapes
:?????????
?
;training/Adam/gradients/loss/ground_loss/truediv_1_grad/SumSum?training/Adam/gradients/loss/ground_loss/truediv_1_grad/RealDivMtraining/Adam/gradients/loss/ground_loss/truediv_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*-
_class#
!loc:@loss/ground_loss/truediv_1*
_output_shapes
:
?
?training/Adam/gradients/loss/ground_loss/truediv_1_grad/ReshapeReshape;training/Adam/gradients/loss/ground_loss/truediv_1_grad/Sum=training/Adam/gradients/loss/ground_loss/truediv_1_grad/Shape*
T0*
Tshape0*-
_class#
!loc:@loss/ground_loss/truediv_1*#
_output_shapes
:?????????
?
;training/Adam/gradients/loss/ground_loss/truediv_1_grad/NegNegloss/ground_loss/mul*
T0*-
_class#
!loc:@loss/ground_loss/truediv_1*#
_output_shapes
:?????????
?
Atraining/Adam/gradients/loss/ground_loss/truediv_1_grad/RealDiv_1RealDiv;training/Adam/gradients/loss/ground_loss/truediv_1_grad/Negloss/ground_loss/Mean_2*#
_output_shapes
:?????????*
T0*-
_class#
!loc:@loss/ground_loss/truediv_1
?
Atraining/Adam/gradients/loss/ground_loss/truediv_1_grad/RealDiv_2RealDivAtraining/Adam/gradients/loss/ground_loss/truediv_1_grad/RealDiv_1loss/ground_loss/Mean_2*#
_output_shapes
:?????????*
T0*-
_class#
!loc:@loss/ground_loss/truediv_1
?
;training/Adam/gradients/loss/ground_loss/truediv_1_grad/mulMul<training/Adam/gradients/loss/ground_loss/Mean_3_grad/truedivAtraining/Adam/gradients/loss/ground_loss/truediv_1_grad/RealDiv_2*#
_output_shapes
:?????????*
T0*-
_class#
!loc:@loss/ground_loss/truediv_1
?
=training/Adam/gradients/loss/ground_loss/truediv_1_grad/Sum_1Sum;training/Adam/gradients/loss/ground_loss/truediv_1_grad/mulOtraining/Adam/gradients/loss/ground_loss/truediv_1_grad/BroadcastGradientArgs:1*
T0*-
_class#
!loc:@loss/ground_loss/truediv_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
Atraining/Adam/gradients/loss/ground_loss/truediv_1_grad/Reshape_1Reshape=training/Adam/gradients/loss/ground_loss/truediv_1_grad/Sum_1?training/Adam/gradients/loss/ground_loss/truediv_1_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0*-
_class#
!loc:@loss/ground_loss/truediv_1
?
:training/Adam/gradients/loss/weather_loss/mul_1_grad/ShapeShapeloss/weather_loss/Mean*
T0*
out_type0**
_class 
loc:@loss/weather_loss/mul_1*
_output_shapes
:
?
<training/Adam/gradients/loss/weather_loss/mul_1_grad/Shape_1Shapeweather_sample_weights*
T0*
out_type0**
_class 
loc:@loss/weather_loss/mul_1*
_output_shapes
:
?
Jtraining/Adam/gradients/loss/weather_loss/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs:training/Adam/gradients/loss/weather_loss/mul_1_grad/Shape<training/Adam/gradients/loss/weather_loss/mul_1_grad/Shape_1*
T0**
_class 
loc:@loss/weather_loss/mul_1*2
_output_shapes 
:?????????:?????????
?
8training/Adam/gradients/loss/weather_loss/mul_1_grad/MulMul@training/Adam/gradients/loss/weather_loss/truediv_1_grad/Reshapeweather_sample_weights*#
_output_shapes
:?????????*
T0**
_class 
loc:@loss/weather_loss/mul_1
?
8training/Adam/gradients/loss/weather_loss/mul_1_grad/SumSum8training/Adam/gradients/loss/weather_loss/mul_1_grad/MulJtraining/Adam/gradients/loss/weather_loss/mul_1_grad/BroadcastGradientArgs*
T0**
_class 
loc:@loss/weather_loss/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
<training/Adam/gradients/loss/weather_loss/mul_1_grad/ReshapeReshape8training/Adam/gradients/loss/weather_loss/mul_1_grad/Sum:training/Adam/gradients/loss/weather_loss/mul_1_grad/Shape*
T0*
Tshape0**
_class 
loc:@loss/weather_loss/mul_1*#
_output_shapes
:?????????
?
:training/Adam/gradients/loss/weather_loss/mul_1_grad/Mul_1Mulloss/weather_loss/Mean@training/Adam/gradients/loss/weather_loss/truediv_1_grad/Reshape*
T0**
_class 
loc:@loss/weather_loss/mul_1*#
_output_shapes
:?????????
?
:training/Adam/gradients/loss/weather_loss/mul_1_grad/Sum_1Sum:training/Adam/gradients/loss/weather_loss/mul_1_grad/Mul_1Ltraining/Adam/gradients/loss/weather_loss/mul_1_grad/BroadcastGradientArgs:1*
T0**
_class 
loc:@loss/weather_loss/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
>training/Adam/gradients/loss/weather_loss/mul_1_grad/Reshape_1Reshape:training/Adam/gradients/loss/weather_loss/mul_1_grad/Sum_1<training/Adam/gradients/loss/weather_loss/mul_1_grad/Shape_1*
T0*
Tshape0**
_class 
loc:@loss/weather_loss/mul_1*#
_output_shapes
:?????????
?
7training/Adam/gradients/loss/ground_loss/mul_grad/ShapeShapeloss/ground_loss/Mean_1*
T0*
out_type0*'
_class
loc:@loss/ground_loss/mul*
_output_shapes
:
?
9training/Adam/gradients/loss/ground_loss/mul_grad/Shape_1Shapeground_sample_weights*
T0*
out_type0*'
_class
loc:@loss/ground_loss/mul*
_output_shapes
:
?
Gtraining/Adam/gradients/loss/ground_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs7training/Adam/gradients/loss/ground_loss/mul_grad/Shape9training/Adam/gradients/loss/ground_loss/mul_grad/Shape_1*
T0*'
_class
loc:@loss/ground_loss/mul*2
_output_shapes 
:?????????:?????????
?
5training/Adam/gradients/loss/ground_loss/mul_grad/MulMul?training/Adam/gradients/loss/ground_loss/truediv_1_grad/Reshapeground_sample_weights*
T0*'
_class
loc:@loss/ground_loss/mul*#
_output_shapes
:?????????
?
5training/Adam/gradients/loss/ground_loss/mul_grad/SumSum5training/Adam/gradients/loss/ground_loss/mul_grad/MulGtraining/Adam/gradients/loss/ground_loss/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*'
_class
loc:@loss/ground_loss/mul*
_output_shapes
:
?
9training/Adam/gradients/loss/ground_loss/mul_grad/ReshapeReshape5training/Adam/gradients/loss/ground_loss/mul_grad/Sum7training/Adam/gradients/loss/ground_loss/mul_grad/Shape*
T0*
Tshape0*'
_class
loc:@loss/ground_loss/mul*#
_output_shapes
:?????????
?
7training/Adam/gradients/loss/ground_loss/mul_grad/Mul_1Mulloss/ground_loss/Mean_1?training/Adam/gradients/loss/ground_loss/truediv_1_grad/Reshape*
T0*'
_class
loc:@loss/ground_loss/mul*#
_output_shapes
:?????????
?
7training/Adam/gradients/loss/ground_loss/mul_grad/Sum_1Sum7training/Adam/gradients/loss/ground_loss/mul_grad/Mul_1Itraining/Adam/gradients/loss/ground_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*'
_class
loc:@loss/ground_loss/mul
?
;training/Adam/gradients/loss/ground_loss/mul_grad/Reshape_1Reshape7training/Adam/gradients/loss/ground_loss/mul_grad/Sum_19training/Adam/gradients/loss/ground_loss/mul_grad/Shape_1*#
_output_shapes
:?????????*
T0*
Tshape0*'
_class
loc:@loss/ground_loss/mul
?
9training/Adam/gradients/loss/weather_loss/Mean_grad/ShapeShapeloss/weather_loss/Neg*
T0*
out_type0*)
_class
loc:@loss/weather_loss/Mean*
_output_shapes
:
?
8training/Adam/gradients/loss/weather_loss/Mean_grad/SizeConst*
value	B :*)
_class
loc:@loss/weather_loss/Mean*
dtype0*
_output_shapes
: 
?
7training/Adam/gradients/loss/weather_loss/Mean_grad/addAdd(loss/weather_loss/Mean/reduction_indices8training/Adam/gradients/loss/weather_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/weather_loss/Mean*
_output_shapes
: 
?
7training/Adam/gradients/loss/weather_loss/Mean_grad/modFloorMod7training/Adam/gradients/loss/weather_loss/Mean_grad/add8training/Adam/gradients/loss/weather_loss/Mean_grad/Size*
_output_shapes
: *
T0*)
_class
loc:@loss/weather_loss/Mean
?
;training/Adam/gradients/loss/weather_loss/Mean_grad/Shape_1Const*
valueB: *)
_class
loc:@loss/weather_loss/Mean*
dtype0*
_output_shapes
:
?
?training/Adam/gradients/loss/weather_loss/Mean_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *)
_class
loc:@loss/weather_loss/Mean
?
?training/Adam/gradients/loss/weather_loss/Mean_grad/range/deltaConst*
value	B :*)
_class
loc:@loss/weather_loss/Mean*
dtype0*
_output_shapes
: 
?
9training/Adam/gradients/loss/weather_loss/Mean_grad/rangeRange?training/Adam/gradients/loss/weather_loss/Mean_grad/range/start8training/Adam/gradients/loss/weather_loss/Mean_grad/Size?training/Adam/gradients/loss/weather_loss/Mean_grad/range/delta*)
_class
loc:@loss/weather_loss/Mean*
_output_shapes
:*

Tidx0
?
>training/Adam/gradients/loss/weather_loss/Mean_grad/Fill/valueConst*
value	B :*)
_class
loc:@loss/weather_loss/Mean*
dtype0*
_output_shapes
: 
?
8training/Adam/gradients/loss/weather_loss/Mean_grad/FillFill;training/Adam/gradients/loss/weather_loss/Mean_grad/Shape_1>training/Adam/gradients/loss/weather_loss/Mean_grad/Fill/value*
T0*

index_type0*)
_class
loc:@loss/weather_loss/Mean*
_output_shapes
: 
?
Atraining/Adam/gradients/loss/weather_loss/Mean_grad/DynamicStitchDynamicStitch9training/Adam/gradients/loss/weather_loss/Mean_grad/range7training/Adam/gradients/loss/weather_loss/Mean_grad/mod9training/Adam/gradients/loss/weather_loss/Mean_grad/Shape8training/Adam/gradients/loss/weather_loss/Mean_grad/Fill*
T0*)
_class
loc:@loss/weather_loss/Mean*
N*#
_output_shapes
:?????????
?
=training/Adam/gradients/loss/weather_loss/Mean_grad/Maximum/yConst*
value	B :*)
_class
loc:@loss/weather_loss/Mean*
dtype0*
_output_shapes
: 
?
;training/Adam/gradients/loss/weather_loss/Mean_grad/MaximumMaximumAtraining/Adam/gradients/loss/weather_loss/Mean_grad/DynamicStitch=training/Adam/gradients/loss/weather_loss/Mean_grad/Maximum/y*
T0*)
_class
loc:@loss/weather_loss/Mean*#
_output_shapes
:?????????
?
<training/Adam/gradients/loss/weather_loss/Mean_grad/floordivFloorDiv9training/Adam/gradients/loss/weather_loss/Mean_grad/Shape;training/Adam/gradients/loss/weather_loss/Mean_grad/Maximum*
T0*)
_class
loc:@loss/weather_loss/Mean*#
_output_shapes
:?????????
?
;training/Adam/gradients/loss/weather_loss/Mean_grad/ReshapeReshape<training/Adam/gradients/loss/weather_loss/mul_1_grad/ReshapeAtraining/Adam/gradients/loss/weather_loss/Mean_grad/DynamicStitch*
T0*
Tshape0*)
_class
loc:@loss/weather_loss/Mean*
_output_shapes
:
?
8training/Adam/gradients/loss/weather_loss/Mean_grad/TileTile;training/Adam/gradients/loss/weather_loss/Mean_grad/Reshape<training/Adam/gradients/loss/weather_loss/Mean_grad/floordiv*

Tmultiples0*
T0*)
_class
loc:@loss/weather_loss/Mean*
_output_shapes
:
?
;training/Adam/gradients/loss/weather_loss/Mean_grad/Shape_2Shapeloss/weather_loss/Neg*
_output_shapes
:*
T0*
out_type0*)
_class
loc:@loss/weather_loss/Mean
?
;training/Adam/gradients/loss/weather_loss/Mean_grad/Shape_3Shapeloss/weather_loss/Mean*
T0*
out_type0*)
_class
loc:@loss/weather_loss/Mean*
_output_shapes
:
?
9training/Adam/gradients/loss/weather_loss/Mean_grad/ConstConst*
valueB: *)
_class
loc:@loss/weather_loss/Mean*
dtype0*
_output_shapes
:
?
8training/Adam/gradients/loss/weather_loss/Mean_grad/ProdProd;training/Adam/gradients/loss/weather_loss/Mean_grad/Shape_29training/Adam/gradients/loss/weather_loss/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0*)
_class
loc:@loss/weather_loss/Mean*
_output_shapes
: 
?
;training/Adam/gradients/loss/weather_loss/Mean_grad/Const_1Const*
valueB: *)
_class
loc:@loss/weather_loss/Mean*
dtype0*
_output_shapes
:
?
:training/Adam/gradients/loss/weather_loss/Mean_grad/Prod_1Prod;training/Adam/gradients/loss/weather_loss/Mean_grad/Shape_3;training/Adam/gradients/loss/weather_loss/Mean_grad/Const_1*
T0*)
_class
loc:@loss/weather_loss/Mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
?
?training/Adam/gradients/loss/weather_loss/Mean_grad/Maximum_1/yConst*
value	B :*)
_class
loc:@loss/weather_loss/Mean*
dtype0*
_output_shapes
: 
?
=training/Adam/gradients/loss/weather_loss/Mean_grad/Maximum_1Maximum:training/Adam/gradients/loss/weather_loss/Mean_grad/Prod_1?training/Adam/gradients/loss/weather_loss/Mean_grad/Maximum_1/y*
_output_shapes
: *
T0*)
_class
loc:@loss/weather_loss/Mean
?
>training/Adam/gradients/loss/weather_loss/Mean_grad/floordiv_1FloorDiv8training/Adam/gradients/loss/weather_loss/Mean_grad/Prod=training/Adam/gradients/loss/weather_loss/Mean_grad/Maximum_1*
_output_shapes
: *
T0*)
_class
loc:@loss/weather_loss/Mean
?
8training/Adam/gradients/loss/weather_loss/Mean_grad/CastCast>training/Adam/gradients/loss/weather_loss/Mean_grad/floordiv_1*

DstT0*
_output_shapes
: *

SrcT0*)
_class
loc:@loss/weather_loss/Mean
?
;training/Adam/gradients/loss/weather_loss/Mean_grad/truedivRealDiv8training/Adam/gradients/loss/weather_loss/Mean_grad/Tile8training/Adam/gradients/loss/weather_loss/Mean_grad/Cast*
T0*)
_class
loc:@loss/weather_loss/Mean*#
_output_shapes
:?????????
?
:training/Adam/gradients/loss/ground_loss/Mean_1_grad/ShapeShapeloss/ground_loss/Mean*
T0*
out_type0**
_class 
loc:@loss/ground_loss/Mean_1*
_output_shapes
:
?
9training/Adam/gradients/loss/ground_loss/Mean_1_grad/SizeConst*
value	B :**
_class 
loc:@loss/ground_loss/Mean_1*
dtype0*
_output_shapes
: 
?
8training/Adam/gradients/loss/ground_loss/Mean_1_grad/addAdd)loss/ground_loss/Mean_1/reduction_indices9training/Adam/gradients/loss/ground_loss/Mean_1_grad/Size*
T0**
_class 
loc:@loss/ground_loss/Mean_1*
_output_shapes
: 
?
8training/Adam/gradients/loss/ground_loss/Mean_1_grad/modFloorMod8training/Adam/gradients/loss/ground_loss/Mean_1_grad/add9training/Adam/gradients/loss/ground_loss/Mean_1_grad/Size*
T0**
_class 
loc:@loss/ground_loss/Mean_1*
_output_shapes
: 
?
<training/Adam/gradients/loss/ground_loss/Mean_1_grad/Shape_1Const*
valueB: **
_class 
loc:@loss/ground_loss/Mean_1*
dtype0*
_output_shapes
:
?
@training/Adam/gradients/loss/ground_loss/Mean_1_grad/range/startConst*
value	B : **
_class 
loc:@loss/ground_loss/Mean_1*
dtype0*
_output_shapes
: 
?
@training/Adam/gradients/loss/ground_loss/Mean_1_grad/range/deltaConst*
value	B :**
_class 
loc:@loss/ground_loss/Mean_1*
dtype0*
_output_shapes
: 
?
:training/Adam/gradients/loss/ground_loss/Mean_1_grad/rangeRange@training/Adam/gradients/loss/ground_loss/Mean_1_grad/range/start9training/Adam/gradients/loss/ground_loss/Mean_1_grad/Size@training/Adam/gradients/loss/ground_loss/Mean_1_grad/range/delta**
_class 
loc:@loss/ground_loss/Mean_1*
_output_shapes
:*

Tidx0
?
?training/Adam/gradients/loss/ground_loss/Mean_1_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :**
_class 
loc:@loss/ground_loss/Mean_1
?
9training/Adam/gradients/loss/ground_loss/Mean_1_grad/FillFill<training/Adam/gradients/loss/ground_loss/Mean_1_grad/Shape_1?training/Adam/gradients/loss/ground_loss/Mean_1_grad/Fill/value*
T0*

index_type0**
_class 
loc:@loss/ground_loss/Mean_1*
_output_shapes
: 
?
Btraining/Adam/gradients/loss/ground_loss/Mean_1_grad/DynamicStitchDynamicStitch:training/Adam/gradients/loss/ground_loss/Mean_1_grad/range8training/Adam/gradients/loss/ground_loss/Mean_1_grad/mod:training/Adam/gradients/loss/ground_loss/Mean_1_grad/Shape9training/Adam/gradients/loss/ground_loss/Mean_1_grad/Fill*
T0**
_class 
loc:@loss/ground_loss/Mean_1*
N*#
_output_shapes
:?????????
?
>training/Adam/gradients/loss/ground_loss/Mean_1_grad/Maximum/yConst*
value	B :**
_class 
loc:@loss/ground_loss/Mean_1*
dtype0*
_output_shapes
: 
?
<training/Adam/gradients/loss/ground_loss/Mean_1_grad/MaximumMaximumBtraining/Adam/gradients/loss/ground_loss/Mean_1_grad/DynamicStitch>training/Adam/gradients/loss/ground_loss/Mean_1_grad/Maximum/y*
T0**
_class 
loc:@loss/ground_loss/Mean_1*#
_output_shapes
:?????????
?
=training/Adam/gradients/loss/ground_loss/Mean_1_grad/floordivFloorDiv:training/Adam/gradients/loss/ground_loss/Mean_1_grad/Shape<training/Adam/gradients/loss/ground_loss/Mean_1_grad/Maximum*
T0**
_class 
loc:@loss/ground_loss/Mean_1*#
_output_shapes
:?????????
?
<training/Adam/gradients/loss/ground_loss/Mean_1_grad/ReshapeReshape9training/Adam/gradients/loss/ground_loss/mul_grad/ReshapeBtraining/Adam/gradients/loss/ground_loss/Mean_1_grad/DynamicStitch*
T0*
Tshape0**
_class 
loc:@loss/ground_loss/Mean_1*
_output_shapes
:
?
9training/Adam/gradients/loss/ground_loss/Mean_1_grad/TileTile<training/Adam/gradients/loss/ground_loss/Mean_1_grad/Reshape=training/Adam/gradients/loss/ground_loss/Mean_1_grad/floordiv*
_output_shapes
:*

Tmultiples0*
T0**
_class 
loc:@loss/ground_loss/Mean_1
?
<training/Adam/gradients/loss/ground_loss/Mean_1_grad/Shape_2Shapeloss/ground_loss/Mean*
T0*
out_type0**
_class 
loc:@loss/ground_loss/Mean_1*
_output_shapes
:
?
<training/Adam/gradients/loss/ground_loss/Mean_1_grad/Shape_3Shapeloss/ground_loss/Mean_1*
T0*
out_type0**
_class 
loc:@loss/ground_loss/Mean_1*
_output_shapes
:
?
:training/Adam/gradients/loss/ground_loss/Mean_1_grad/ConstConst*
valueB: **
_class 
loc:@loss/ground_loss/Mean_1*
dtype0*
_output_shapes
:
?
9training/Adam/gradients/loss/ground_loss/Mean_1_grad/ProdProd<training/Adam/gradients/loss/ground_loss/Mean_1_grad/Shape_2:training/Adam/gradients/loss/ground_loss/Mean_1_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0**
_class 
loc:@loss/ground_loss/Mean_1
?
<training/Adam/gradients/loss/ground_loss/Mean_1_grad/Const_1Const*
valueB: **
_class 
loc:@loss/ground_loss/Mean_1*
dtype0*
_output_shapes
:
?
;training/Adam/gradients/loss/ground_loss/Mean_1_grad/Prod_1Prod<training/Adam/gradients/loss/ground_loss/Mean_1_grad/Shape_3<training/Adam/gradients/loss/ground_loss/Mean_1_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0**
_class 
loc:@loss/ground_loss/Mean_1
?
@training/Adam/gradients/loss/ground_loss/Mean_1_grad/Maximum_1/yConst*
value	B :**
_class 
loc:@loss/ground_loss/Mean_1*
dtype0*
_output_shapes
: 
?
>training/Adam/gradients/loss/ground_loss/Mean_1_grad/Maximum_1Maximum;training/Adam/gradients/loss/ground_loss/Mean_1_grad/Prod_1@training/Adam/gradients/loss/ground_loss/Mean_1_grad/Maximum_1/y*
T0**
_class 
loc:@loss/ground_loss/Mean_1*
_output_shapes
: 
?
?training/Adam/gradients/loss/ground_loss/Mean_1_grad/floordiv_1FloorDiv9training/Adam/gradients/loss/ground_loss/Mean_1_grad/Prod>training/Adam/gradients/loss/ground_loss/Mean_1_grad/Maximum_1*
T0**
_class 
loc:@loss/ground_loss/Mean_1*
_output_shapes
: 
?
9training/Adam/gradients/loss/ground_loss/Mean_1_grad/CastCast?training/Adam/gradients/loss/ground_loss/Mean_1_grad/floordiv_1*

SrcT0**
_class 
loc:@loss/ground_loss/Mean_1*

DstT0*
_output_shapes
: 
?
<training/Adam/gradients/loss/ground_loss/Mean_1_grad/truedivRealDiv9training/Adam/gradients/loss/ground_loss/Mean_1_grad/Tile9training/Adam/gradients/loss/ground_loss/Mean_1_grad/Cast*
T0**
_class 
loc:@loss/ground_loss/Mean_1*#
_output_shapes
:?????????
?
6training/Adam/gradients/loss/weather_loss/Neg_grad/NegNeg;training/Adam/gradients/loss/weather_loss/Mean_grad/truediv*
T0*(
_class
loc:@loss/weather_loss/Neg*#
_output_shapes
:?????????
?
8training/Adam/gradients/loss/ground_loss/Mean_grad/ShapeShapeloss/ground_loss/logistic_loss*
T0*
out_type0*(
_class
loc:@loss/ground_loss/Mean*
_output_shapes
:
?
7training/Adam/gradients/loss/ground_loss/Mean_grad/SizeConst*
value	B :*(
_class
loc:@loss/ground_loss/Mean*
dtype0*
_output_shapes
: 
?
6training/Adam/gradients/loss/ground_loss/Mean_grad/addAdd'loss/ground_loss/Mean/reduction_indices7training/Adam/gradients/loss/ground_loss/Mean_grad/Size*
T0*(
_class
loc:@loss/ground_loss/Mean*
_output_shapes
: 
?
6training/Adam/gradients/loss/ground_loss/Mean_grad/modFloorMod6training/Adam/gradients/loss/ground_loss/Mean_grad/add7training/Adam/gradients/loss/ground_loss/Mean_grad/Size*
T0*(
_class
loc:@loss/ground_loss/Mean*
_output_shapes
: 
?
:training/Adam/gradients/loss/ground_loss/Mean_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *(
_class
loc:@loss/ground_loss/Mean
?
>training/Adam/gradients/loss/ground_loss/Mean_grad/range/startConst*
value	B : *(
_class
loc:@loss/ground_loss/Mean*
dtype0*
_output_shapes
: 
?
>training/Adam/gradients/loss/ground_loss/Mean_grad/range/deltaConst*
value	B :*(
_class
loc:@loss/ground_loss/Mean*
dtype0*
_output_shapes
: 
?
8training/Adam/gradients/loss/ground_loss/Mean_grad/rangeRange>training/Adam/gradients/loss/ground_loss/Mean_grad/range/start7training/Adam/gradients/loss/ground_loss/Mean_grad/Size>training/Adam/gradients/loss/ground_loss/Mean_grad/range/delta*

Tidx0*(
_class
loc:@loss/ground_loss/Mean*
_output_shapes
:
?
=training/Adam/gradients/loss/ground_loss/Mean_grad/Fill/valueConst*
value	B :*(
_class
loc:@loss/ground_loss/Mean*
dtype0*
_output_shapes
: 
?
7training/Adam/gradients/loss/ground_loss/Mean_grad/FillFill:training/Adam/gradients/loss/ground_loss/Mean_grad/Shape_1=training/Adam/gradients/loss/ground_loss/Mean_grad/Fill/value*
T0*

index_type0*(
_class
loc:@loss/ground_loss/Mean*
_output_shapes
: 
?
@training/Adam/gradients/loss/ground_loss/Mean_grad/DynamicStitchDynamicStitch8training/Adam/gradients/loss/ground_loss/Mean_grad/range6training/Adam/gradients/loss/ground_loss/Mean_grad/mod8training/Adam/gradients/loss/ground_loss/Mean_grad/Shape7training/Adam/gradients/loss/ground_loss/Mean_grad/Fill*
T0*(
_class
loc:@loss/ground_loss/Mean*
N*#
_output_shapes
:?????????
?
<training/Adam/gradients/loss/ground_loss/Mean_grad/Maximum/yConst*
value	B :*(
_class
loc:@loss/ground_loss/Mean*
dtype0*
_output_shapes
: 
?
:training/Adam/gradients/loss/ground_loss/Mean_grad/MaximumMaximum@training/Adam/gradients/loss/ground_loss/Mean_grad/DynamicStitch<training/Adam/gradients/loss/ground_loss/Mean_grad/Maximum/y*#
_output_shapes
:?????????*
T0*(
_class
loc:@loss/ground_loss/Mean
?
;training/Adam/gradients/loss/ground_loss/Mean_grad/floordivFloorDiv8training/Adam/gradients/loss/ground_loss/Mean_grad/Shape:training/Adam/gradients/loss/ground_loss/Mean_grad/Maximum*
T0*(
_class
loc:@loss/ground_loss/Mean*
_output_shapes
:
?
:training/Adam/gradients/loss/ground_loss/Mean_grad/ReshapeReshape<training/Adam/gradients/loss/ground_loss/Mean_1_grad/truediv@training/Adam/gradients/loss/ground_loss/Mean_grad/DynamicStitch*
T0*
Tshape0*(
_class
loc:@loss/ground_loss/Mean*
_output_shapes
:
?
7training/Adam/gradients/loss/ground_loss/Mean_grad/TileTile:training/Adam/gradients/loss/ground_loss/Mean_grad/Reshape;training/Adam/gradients/loss/ground_loss/Mean_grad/floordiv*0
_output_shapes
:??????????????????*

Tmultiples0*
T0*(
_class
loc:@loss/ground_loss/Mean
?
:training/Adam/gradients/loss/ground_loss/Mean_grad/Shape_2Shapeloss/ground_loss/logistic_loss*
T0*
out_type0*(
_class
loc:@loss/ground_loss/Mean*
_output_shapes
:
?
:training/Adam/gradients/loss/ground_loss/Mean_grad/Shape_3Shapeloss/ground_loss/Mean*
T0*
out_type0*(
_class
loc:@loss/ground_loss/Mean*
_output_shapes
:
?
8training/Adam/gradients/loss/ground_loss/Mean_grad/ConstConst*
valueB: *(
_class
loc:@loss/ground_loss/Mean*
dtype0*
_output_shapes
:
?
7training/Adam/gradients/loss/ground_loss/Mean_grad/ProdProd:training/Adam/gradients/loss/ground_loss/Mean_grad/Shape_28training/Adam/gradients/loss/ground_loss/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0*(
_class
loc:@loss/ground_loss/Mean*
_output_shapes
: 
?
:training/Adam/gradients/loss/ground_loss/Mean_grad/Const_1Const*
valueB: *(
_class
loc:@loss/ground_loss/Mean*
dtype0*
_output_shapes
:
?
9training/Adam/gradients/loss/ground_loss/Mean_grad/Prod_1Prod:training/Adam/gradients/loss/ground_loss/Mean_grad/Shape_3:training/Adam/gradients/loss/ground_loss/Mean_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*(
_class
loc:@loss/ground_loss/Mean
?
>training/Adam/gradients/loss/ground_loss/Mean_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :*(
_class
loc:@loss/ground_loss/Mean
?
<training/Adam/gradients/loss/ground_loss/Mean_grad/Maximum_1Maximum9training/Adam/gradients/loss/ground_loss/Mean_grad/Prod_1>training/Adam/gradients/loss/ground_loss/Mean_grad/Maximum_1/y*
T0*(
_class
loc:@loss/ground_loss/Mean*
_output_shapes
: 
?
=training/Adam/gradients/loss/ground_loss/Mean_grad/floordiv_1FloorDiv7training/Adam/gradients/loss/ground_loss/Mean_grad/Prod<training/Adam/gradients/loss/ground_loss/Mean_grad/Maximum_1*
T0*(
_class
loc:@loss/ground_loss/Mean*
_output_shapes
: 
?
7training/Adam/gradients/loss/ground_loss/Mean_grad/CastCast=training/Adam/gradients/loss/ground_loss/Mean_grad/floordiv_1*

SrcT0*(
_class
loc:@loss/ground_loss/Mean*

DstT0*
_output_shapes
: 
?
:training/Adam/gradients/loss/ground_loss/Mean_grad/truedivRealDiv7training/Adam/gradients/loss/ground_loss/Mean_grad/Tile7training/Adam/gradients/loss/ground_loss/Mean_grad/Cast*'
_output_shapes
:?????????*
T0*(
_class
loc:@loss/ground_loss/Mean
?
:training/Adam/gradients/loss/weather_loss/Sum_1_grad/ShapeShapeloss/weather_loss/mul*
_output_shapes
:*
T0*
out_type0**
_class 
loc:@loss/weather_loss/Sum_1
?
9training/Adam/gradients/loss/weather_loss/Sum_1_grad/SizeConst*
value	B :**
_class 
loc:@loss/weather_loss/Sum_1*
dtype0*
_output_shapes
: 
?
8training/Adam/gradients/loss/weather_loss/Sum_1_grad/addAdd)loss/weather_loss/Sum_1/reduction_indices9training/Adam/gradients/loss/weather_loss/Sum_1_grad/Size*
_output_shapes
: *
T0**
_class 
loc:@loss/weather_loss/Sum_1
?
8training/Adam/gradients/loss/weather_loss/Sum_1_grad/modFloorMod8training/Adam/gradients/loss/weather_loss/Sum_1_grad/add9training/Adam/gradients/loss/weather_loss/Sum_1_grad/Size*
T0**
_class 
loc:@loss/weather_loss/Sum_1*
_output_shapes
: 
?
<training/Adam/gradients/loss/weather_loss/Sum_1_grad/Shape_1Const*
valueB **
_class 
loc:@loss/weather_loss/Sum_1*
dtype0*
_output_shapes
: 
?
@training/Adam/gradients/loss/weather_loss/Sum_1_grad/range/startConst*
value	B : **
_class 
loc:@loss/weather_loss/Sum_1*
dtype0*
_output_shapes
: 
?
@training/Adam/gradients/loss/weather_loss/Sum_1_grad/range/deltaConst*
value	B :**
_class 
loc:@loss/weather_loss/Sum_1*
dtype0*
_output_shapes
: 
?
:training/Adam/gradients/loss/weather_loss/Sum_1_grad/rangeRange@training/Adam/gradients/loss/weather_loss/Sum_1_grad/range/start9training/Adam/gradients/loss/weather_loss/Sum_1_grad/Size@training/Adam/gradients/loss/weather_loss/Sum_1_grad/range/delta**
_class 
loc:@loss/weather_loss/Sum_1*
_output_shapes
:*

Tidx0
?
?training/Adam/gradients/loss/weather_loss/Sum_1_grad/Fill/valueConst*
value	B :**
_class 
loc:@loss/weather_loss/Sum_1*
dtype0*
_output_shapes
: 
?
9training/Adam/gradients/loss/weather_loss/Sum_1_grad/FillFill<training/Adam/gradients/loss/weather_loss/Sum_1_grad/Shape_1?training/Adam/gradients/loss/weather_loss/Sum_1_grad/Fill/value*
T0*

index_type0**
_class 
loc:@loss/weather_loss/Sum_1*
_output_shapes
: 
?
Btraining/Adam/gradients/loss/weather_loss/Sum_1_grad/DynamicStitchDynamicStitch:training/Adam/gradients/loss/weather_loss/Sum_1_grad/range8training/Adam/gradients/loss/weather_loss/Sum_1_grad/mod:training/Adam/gradients/loss/weather_loss/Sum_1_grad/Shape9training/Adam/gradients/loss/weather_loss/Sum_1_grad/Fill*
T0**
_class 
loc:@loss/weather_loss/Sum_1*
N*#
_output_shapes
:?????????
?
>training/Adam/gradients/loss/weather_loss/Sum_1_grad/Maximum/yConst*
value	B :**
_class 
loc:@loss/weather_loss/Sum_1*
dtype0*
_output_shapes
: 
?
<training/Adam/gradients/loss/weather_loss/Sum_1_grad/MaximumMaximumBtraining/Adam/gradients/loss/weather_loss/Sum_1_grad/DynamicStitch>training/Adam/gradients/loss/weather_loss/Sum_1_grad/Maximum/y*#
_output_shapes
:?????????*
T0**
_class 
loc:@loss/weather_loss/Sum_1
?
=training/Adam/gradients/loss/weather_loss/Sum_1_grad/floordivFloorDiv:training/Adam/gradients/loss/weather_loss/Sum_1_grad/Shape<training/Adam/gradients/loss/weather_loss/Sum_1_grad/Maximum*
T0**
_class 
loc:@loss/weather_loss/Sum_1*
_output_shapes
:
?
<training/Adam/gradients/loss/weather_loss/Sum_1_grad/ReshapeReshape6training/Adam/gradients/loss/weather_loss/Neg_grad/NegBtraining/Adam/gradients/loss/weather_loss/Sum_1_grad/DynamicStitch*
T0*
Tshape0**
_class 
loc:@loss/weather_loss/Sum_1*
_output_shapes
:
?
9training/Adam/gradients/loss/weather_loss/Sum_1_grad/TileTile<training/Adam/gradients/loss/weather_loss/Sum_1_grad/Reshape=training/Adam/gradients/loss/weather_loss/Sum_1_grad/floordiv*
T0**
_class 
loc:@loss/weather_loss/Sum_1*'
_output_shapes
:?????????*

Tmultiples0
?
Atraining/Adam/gradients/loss/ground_loss/logistic_loss_grad/ShapeShape"loss/ground_loss/logistic_loss/sub*
T0*
out_type0*1
_class'
%#loc:@loss/ground_loss/logistic_loss*
_output_shapes
:
?
Ctraining/Adam/gradients/loss/ground_loss/logistic_loss_grad/Shape_1Shape$loss/ground_loss/logistic_loss/Log1p*
T0*
out_type0*1
_class'
%#loc:@loss/ground_loss/logistic_loss*
_output_shapes
:
?
Qtraining/Adam/gradients/loss/ground_loss/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgsAtraining/Adam/gradients/loss/ground_loss/logistic_loss_grad/ShapeCtraining/Adam/gradients/loss/ground_loss/logistic_loss_grad/Shape_1*
T0*1
_class'
%#loc:@loss/ground_loss/logistic_loss*2
_output_shapes 
:?????????:?????????
?
?training/Adam/gradients/loss/ground_loss/logistic_loss_grad/SumSum:training/Adam/gradients/loss/ground_loss/Mean_grad/truedivQtraining/Adam/gradients/loss/ground_loss/logistic_loss_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*1
_class'
%#loc:@loss/ground_loss/logistic_loss*
_output_shapes
:
?
Ctraining/Adam/gradients/loss/ground_loss/logistic_loss_grad/ReshapeReshape?training/Adam/gradients/loss/ground_loss/logistic_loss_grad/SumAtraining/Adam/gradients/loss/ground_loss/logistic_loss_grad/Shape*
T0*
Tshape0*1
_class'
%#loc:@loss/ground_loss/logistic_loss*'
_output_shapes
:?????????
?
Atraining/Adam/gradients/loss/ground_loss/logistic_loss_grad/Sum_1Sum:training/Adam/gradients/loss/ground_loss/Mean_grad/truedivStraining/Adam/gradients/loss/ground_loss/logistic_loss_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*1
_class'
%#loc:@loss/ground_loss/logistic_loss*
_output_shapes
:
?
Etraining/Adam/gradients/loss/ground_loss/logistic_loss_grad/Reshape_1ReshapeAtraining/Adam/gradients/loss/ground_loss/logistic_loss_grad/Sum_1Ctraining/Adam/gradients/loss/ground_loss/logistic_loss_grad/Shape_1*'
_output_shapes
:?????????*
T0*
Tshape0*1
_class'
%#loc:@loss/ground_loss/logistic_loss
?
8training/Adam/gradients/loss/weather_loss/mul_grad/ShapeShapeweather_target*
T0*
out_type0*(
_class
loc:@loss/weather_loss/mul*
_output_shapes
:
?
:training/Adam/gradients/loss/weather_loss/mul_grad/Shape_1Shapeloss/weather_loss/Log*
_output_shapes
:*
T0*
out_type0*(
_class
loc:@loss/weather_loss/mul
?
Htraining/Adam/gradients/loss/weather_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8training/Adam/gradients/loss/weather_loss/mul_grad/Shape:training/Adam/gradients/loss/weather_loss/mul_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0*(
_class
loc:@loss/weather_loss/mul
?
6training/Adam/gradients/loss/weather_loss/mul_grad/MulMul9training/Adam/gradients/loss/weather_loss/Sum_1_grad/Tileloss/weather_loss/Log*
T0*(
_class
loc:@loss/weather_loss/mul*'
_output_shapes
:?????????
?
6training/Adam/gradients/loss/weather_loss/mul_grad/SumSum6training/Adam/gradients/loss/weather_loss/mul_grad/MulHtraining/Adam/gradients/loss/weather_loss/mul_grad/BroadcastGradientArgs*
T0*(
_class
loc:@loss/weather_loss/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
:training/Adam/gradients/loss/weather_loss/mul_grad/ReshapeReshape6training/Adam/gradients/loss/weather_loss/mul_grad/Sum8training/Adam/gradients/loss/weather_loss/mul_grad/Shape*0
_output_shapes
:??????????????????*
T0*
Tshape0*(
_class
loc:@loss/weather_loss/mul
?
8training/Adam/gradients/loss/weather_loss/mul_grad/Mul_1Mulweather_target9training/Adam/gradients/loss/weather_loss/Sum_1_grad/Tile*'
_output_shapes
:?????????*
T0*(
_class
loc:@loss/weather_loss/mul
?
8training/Adam/gradients/loss/weather_loss/mul_grad/Sum_1Sum8training/Adam/gradients/loss/weather_loss/mul_grad/Mul_1Jtraining/Adam/gradients/loss/weather_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*(
_class
loc:@loss/weather_loss/mul
?
<training/Adam/gradients/loss/weather_loss/mul_grad/Reshape_1Reshape8training/Adam/gradients/loss/weather_loss/mul_grad/Sum_1:training/Adam/gradients/loss/weather_loss/mul_grad/Shape_1*'
_output_shapes
:?????????*
T0*
Tshape0*(
_class
loc:@loss/weather_loss/mul
?
Etraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/ShapeShape%loss/ground_loss/logistic_loss/Select*
T0*
out_type0*5
_class+
)'loc:@loss/ground_loss/logistic_loss/sub*
_output_shapes
:
?
Gtraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/Shape_1Shape"loss/ground_loss/logistic_loss/mul*
_output_shapes
:*
T0*
out_type0*5
_class+
)'loc:@loss/ground_loss/logistic_loss/sub
?
Utraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgsEtraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/ShapeGtraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/Shape_1*
T0*5
_class+
)'loc:@loss/ground_loss/logistic_loss/sub*2
_output_shapes 
:?????????:?????????
?
Ctraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/SumSumCtraining/Adam/gradients/loss/ground_loss/logistic_loss_grad/ReshapeUtraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/BroadcastGradientArgs*
T0*5
_class+
)'loc:@loss/ground_loss/logistic_loss/sub*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
Gtraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/ReshapeReshapeCtraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/SumEtraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/Shape*
T0*
Tshape0*5
_class+
)'loc:@loss/ground_loss/logistic_loss/sub*'
_output_shapes
:?????????
?
Etraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/Sum_1SumCtraining/Adam/gradients/loss/ground_loss/logistic_loss_grad/ReshapeWtraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/BroadcastGradientArgs:1*
T0*5
_class+
)'loc:@loss/ground_loss/logistic_loss/sub*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
Ctraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/NegNegEtraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/Sum_1*
_output_shapes
:*
T0*5
_class+
)'loc:@loss/ground_loss/logistic_loss/sub
?
Itraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/Reshape_1ReshapeCtraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/NegGtraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/Shape_1*'
_output_shapes
:?????????*
T0*
Tshape0*5
_class+
)'loc:@loss/ground_loss/logistic_loss/sub
?
Gtraining/Adam/gradients/loss/ground_loss/logistic_loss/Log1p_grad/add/xConstF^training/Adam/gradients/loss/ground_loss/logistic_loss_grad/Reshape_1*
valueB
 *  ??*7
_class-
+)loc:@loss/ground_loss/logistic_loss/Log1p*
dtype0*
_output_shapes
: 
?
Etraining/Adam/gradients/loss/ground_loss/logistic_loss/Log1p_grad/addAddGtraining/Adam/gradients/loss/ground_loss/logistic_loss/Log1p_grad/add/x"loss/ground_loss/logistic_loss/Exp*'
_output_shapes
:?????????*
T0*7
_class-
+)loc:@loss/ground_loss/logistic_loss/Log1p
?
Ltraining/Adam/gradients/loss/ground_loss/logistic_loss/Log1p_grad/Reciprocal
ReciprocalEtraining/Adam/gradients/loss/ground_loss/logistic_loss/Log1p_grad/add*
T0*7
_class-
+)loc:@loss/ground_loss/logistic_loss/Log1p*'
_output_shapes
:?????????
?
Etraining/Adam/gradients/loss/ground_loss/logistic_loss/Log1p_grad/mulMulEtraining/Adam/gradients/loss/ground_loss/logistic_loss_grad/Reshape_1Ltraining/Adam/gradients/loss/ground_loss/logistic_loss/Log1p_grad/Reciprocal*'
_output_shapes
:?????????*
T0*7
_class-
+)loc:@loss/ground_loss/logistic_loss/Log1p
?
=training/Adam/gradients/loss/weather_loss/Log_grad/Reciprocal
Reciprocalloss/weather_loss/clip_by_value=^training/Adam/gradients/loss/weather_loss/mul_grad/Reshape_1*
T0*(
_class
loc:@loss/weather_loss/Log*'
_output_shapes
:?????????
?
6training/Adam/gradients/loss/weather_loss/Log_grad/mulMul<training/Adam/gradients/loss/weather_loss/mul_grad/Reshape_1=training/Adam/gradients/loss/weather_loss/Log_grad/Reciprocal*
T0*(
_class
loc:@loss/weather_loss/Log*'
_output_shapes
:?????????
?
Mtraining/Adam/gradients/loss/ground_loss/logistic_loss/Select_grad/zeros_like	ZerosLikeloss/ground_loss/Log*
T0*8
_class.
,*loc:@loss/ground_loss/logistic_loss/Select*'
_output_shapes
:?????????
?
Itraining/Adam/gradients/loss/ground_loss/logistic_loss/Select_grad/SelectSelect+loss/ground_loss/logistic_loss/GreaterEqualGtraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/ReshapeMtraining/Adam/gradients/loss/ground_loss/logistic_loss/Select_grad/zeros_like*
T0*8
_class.
,*loc:@loss/ground_loss/logistic_loss/Select*'
_output_shapes
:?????????
?
Ktraining/Adam/gradients/loss/ground_loss/logistic_loss/Select_grad/Select_1Select+loss/ground_loss/logistic_loss/GreaterEqualMtraining/Adam/gradients/loss/ground_loss/logistic_loss/Select_grad/zeros_likeGtraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/Reshape*
T0*8
_class.
,*loc:@loss/ground_loss/logistic_loss/Select*'
_output_shapes
:?????????
?
Etraining/Adam/gradients/loss/ground_loss/logistic_loss/mul_grad/ShapeShapeloss/ground_loss/Log*
T0*
out_type0*5
_class+
)'loc:@loss/ground_loss/logistic_loss/mul*
_output_shapes
:
?
Gtraining/Adam/gradients/loss/ground_loss/logistic_loss/mul_grad/Shape_1Shapeground_target*
T0*
out_type0*5
_class+
)'loc:@loss/ground_loss/logistic_loss/mul*
_output_shapes
:
?
Utraining/Adam/gradients/loss/ground_loss/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgsEtraining/Adam/gradients/loss/ground_loss/logistic_loss/mul_grad/ShapeGtraining/Adam/gradients/loss/ground_loss/logistic_loss/mul_grad/Shape_1*
T0*5
_class+
)'loc:@loss/ground_loss/logistic_loss/mul*2
_output_shapes 
:?????????:?????????
?
Ctraining/Adam/gradients/loss/ground_loss/logistic_loss/mul_grad/MulMulItraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/Reshape_1ground_target*
T0*5
_class+
)'loc:@loss/ground_loss/logistic_loss/mul*'
_output_shapes
:?????????
?
Ctraining/Adam/gradients/loss/ground_loss/logistic_loss/mul_grad/SumSumCtraining/Adam/gradients/loss/ground_loss/logistic_loss/mul_grad/MulUtraining/Adam/gradients/loss/ground_loss/logistic_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*5
_class+
)'loc:@loss/ground_loss/logistic_loss/mul
?
Gtraining/Adam/gradients/loss/ground_loss/logistic_loss/mul_grad/ReshapeReshapeCtraining/Adam/gradients/loss/ground_loss/logistic_loss/mul_grad/SumEtraining/Adam/gradients/loss/ground_loss/logistic_loss/mul_grad/Shape*
T0*
Tshape0*5
_class+
)'loc:@loss/ground_loss/logistic_loss/mul*'
_output_shapes
:?????????
?
Etraining/Adam/gradients/loss/ground_loss/logistic_loss/mul_grad/Mul_1Mulloss/ground_loss/LogItraining/Adam/gradients/loss/ground_loss/logistic_loss/sub_grad/Reshape_1*
T0*5
_class+
)'loc:@loss/ground_loss/logistic_loss/mul*'
_output_shapes
:?????????
?
Etraining/Adam/gradients/loss/ground_loss/logistic_loss/mul_grad/Sum_1SumEtraining/Adam/gradients/loss/ground_loss/logistic_loss/mul_grad/Mul_1Wtraining/Adam/gradients/loss/ground_loss/logistic_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*5
_class+
)'loc:@loss/ground_loss/logistic_loss/mul
?
Itraining/Adam/gradients/loss/ground_loss/logistic_loss/mul_grad/Reshape_1ReshapeEtraining/Adam/gradients/loss/ground_loss/logistic_loss/mul_grad/Sum_1Gtraining/Adam/gradients/loss/ground_loss/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0*5
_class+
)'loc:@loss/ground_loss/logistic_loss/mul*0
_output_shapes
:??????????????????
?
Ctraining/Adam/gradients/loss/ground_loss/logistic_loss/Exp_grad/mulMulEtraining/Adam/gradients/loss/ground_loss/logistic_loss/Log1p_grad/mul"loss/ground_loss/logistic_loss/Exp*
T0*5
_class+
)'loc:@loss/ground_loss/logistic_loss/Exp*'
_output_shapes
:?????????
?
Btraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/ShapeShape'loss/weather_loss/clip_by_value/Minimum*
_output_shapes
:*
T0*
out_type0*2
_class(
&$loc:@loss/weather_loss/clip_by_value
?
Dtraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *2
_class(
&$loc:@loss/weather_loss/clip_by_value
?
Dtraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/Shape_2Shape6training/Adam/gradients/loss/weather_loss/Log_grad/mul*
_output_shapes
:*
T0*
out_type0*2
_class(
&$loc:@loss/weather_loss/clip_by_value
?
Htraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *2
_class(
&$loc:@loss/weather_loss/clip_by_value
?
Btraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/zerosFillDtraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/Shape_2Htraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/zeros/Const*
T0*

index_type0*2
_class(
&$loc:@loss/weather_loss/clip_by_value*'
_output_shapes
:?????????
?
Itraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/GreaterEqualGreaterEqual'loss/weather_loss/clip_by_value/Minimumloss/weather_loss/Const*'
_output_shapes
:?????????*
T0*2
_class(
&$loc:@loss/weather_loss/clip_by_value
?
Rtraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsBtraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/ShapeDtraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0*2
_class(
&$loc:@loss/weather_loss/clip_by_value
?
Ctraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/SelectSelectItraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/GreaterEqual6training/Adam/gradients/loss/weather_loss/Log_grad/mulBtraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/zeros*
T0*2
_class(
&$loc:@loss/weather_loss/clip_by_value*'
_output_shapes
:?????????
?
Etraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/Select_1SelectItraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/GreaterEqualBtraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/zeros6training/Adam/gradients/loss/weather_loss/Log_grad/mul*
T0*2
_class(
&$loc:@loss/weather_loss/clip_by_value*'
_output_shapes
:?????????
?
@training/Adam/gradients/loss/weather_loss/clip_by_value_grad/SumSumCtraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/SelectRtraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/BroadcastGradientArgs*
T0*2
_class(
&$loc:@loss/weather_loss/clip_by_value*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
Dtraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/ReshapeReshape@training/Adam/gradients/loss/weather_loss/clip_by_value_grad/SumBtraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/Shape*'
_output_shapes
:?????????*
T0*
Tshape0*2
_class(
&$loc:@loss/weather_loss/clip_by_value
?
Btraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/Sum_1SumEtraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/Select_1Ttraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/BroadcastGradientArgs:1*2
_class(
&$loc:@loss/weather_loss/clip_by_value*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
?
Ftraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/Reshape_1ReshapeBtraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/Sum_1Dtraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/Shape_1*
T0*
Tshape0*2
_class(
&$loc:@loss/weather_loss/clip_by_value*
_output_shapes
: 
?
Otraining/Adam/gradients/loss/ground_loss/logistic_loss/Select_1_grad/zeros_like	ZerosLike"loss/ground_loss/logistic_loss/Neg*'
_output_shapes
:?????????*
T0*:
_class0
.,loc:@loss/ground_loss/logistic_loss/Select_1
?
Ktraining/Adam/gradients/loss/ground_loss/logistic_loss/Select_1_grad/SelectSelect+loss/ground_loss/logistic_loss/GreaterEqualCtraining/Adam/gradients/loss/ground_loss/logistic_loss/Exp_grad/mulOtraining/Adam/gradients/loss/ground_loss/logistic_loss/Select_1_grad/zeros_like*
T0*:
_class0
.,loc:@loss/ground_loss/logistic_loss/Select_1*'
_output_shapes
:?????????
?
Mtraining/Adam/gradients/loss/ground_loss/logistic_loss/Select_1_grad/Select_1Select+loss/ground_loss/logistic_loss/GreaterEqualOtraining/Adam/gradients/loss/ground_loss/logistic_loss/Select_1_grad/zeros_likeCtraining/Adam/gradients/loss/ground_loss/logistic_loss/Exp_grad/mul*
T0*:
_class0
.,loc:@loss/ground_loss/logistic_loss/Select_1*'
_output_shapes
:?????????
?
Jtraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/ShapeShapeloss/weather_loss/truediv*
T0*
out_type0*:
_class0
.,loc:@loss/weather_loss/clip_by_value/Minimum*
_output_shapes
:
?
Ltraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *:
_class0
.,loc:@loss/weather_loss/clip_by_value/Minimum
?
Ltraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/Shape_2ShapeDtraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/Reshape*
_output_shapes
:*
T0*
out_type0*:
_class0
.,loc:@loss/weather_loss/clip_by_value/Minimum
?
Ptraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *:
_class0
.,loc:@loss/weather_loss/clip_by_value/Minimum
?
Jtraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/zerosFillLtraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/Shape_2Ptraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/zeros/Const*'
_output_shapes
:?????????*
T0*

index_type0*:
_class0
.,loc:@loss/weather_loss/clip_by_value/Minimum
?
Ntraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/LessEqual	LessEqualloss/weather_loss/truedivloss/weather_loss/sub*'
_output_shapes
:?????????*
T0*:
_class0
.,loc:@loss/weather_loss/clip_by_value/Minimum
?
Ztraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsJtraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/ShapeLtraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/Shape_1*
T0*:
_class0
.,loc:@loss/weather_loss/clip_by_value/Minimum*2
_output_shapes 
:?????????:?????????
?
Ktraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/SelectSelectNtraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/LessEqualDtraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/ReshapeJtraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/zeros*'
_output_shapes
:?????????*
T0*:
_class0
.,loc:@loss/weather_loss/clip_by_value/Minimum
?
Mtraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/Select_1SelectNtraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/LessEqualJtraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/zerosDtraining/Adam/gradients/loss/weather_loss/clip_by_value_grad/Reshape*
T0*:
_class0
.,loc:@loss/weather_loss/clip_by_value/Minimum*'
_output_shapes
:?????????
?
Htraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/SumSumKtraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/SelectZtraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*
T0*:
_class0
.,loc:@loss/weather_loss/clip_by_value/Minimum*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
Ltraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/ReshapeReshapeHtraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/SumJtraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/Shape*
T0*
Tshape0*:
_class0
.,loc:@loss/weather_loss/clip_by_value/Minimum*'
_output_shapes
:?????????
?
Jtraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/Sum_1SumMtraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/Select_1\training/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
T0*:
_class0
.,loc:@loss/weather_loss/clip_by_value/Minimum*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
Ntraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/Reshape_1ReshapeJtraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/Sum_1Ltraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0*:
_class0
.,loc:@loss/weather_loss/clip_by_value/Minimum
?
Ctraining/Adam/gradients/loss/ground_loss/logistic_loss/Neg_grad/NegNegKtraining/Adam/gradients/loss/ground_loss/logistic_loss/Select_1_grad/Select*
T0*5
_class+
)'loc:@loss/ground_loss/logistic_loss/Neg*'
_output_shapes
:?????????
?
<training/Adam/gradients/loss/weather_loss/truediv_grad/ShapeShapeweather/Softmax*
_output_shapes
:*
T0*
out_type0*,
_class"
 loc:@loss/weather_loss/truediv
?
>training/Adam/gradients/loss/weather_loss/truediv_grad/Shape_1Shapeloss/weather_loss/Sum*
T0*
out_type0*,
_class"
 loc:@loss/weather_loss/truediv*
_output_shapes
:
?
Ltraining/Adam/gradients/loss/weather_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs<training/Adam/gradients/loss/weather_loss/truediv_grad/Shape>training/Adam/gradients/loss/weather_loss/truediv_grad/Shape_1*
T0*,
_class"
 loc:@loss/weather_loss/truediv*2
_output_shapes 
:?????????:?????????
?
>training/Adam/gradients/loss/weather_loss/truediv_grad/RealDivRealDivLtraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/Reshapeloss/weather_loss/Sum*
T0*,
_class"
 loc:@loss/weather_loss/truediv*'
_output_shapes
:?????????
?
:training/Adam/gradients/loss/weather_loss/truediv_grad/SumSum>training/Adam/gradients/loss/weather_loss/truediv_grad/RealDivLtraining/Adam/gradients/loss/weather_loss/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*,
_class"
 loc:@loss/weather_loss/truediv
?
>training/Adam/gradients/loss/weather_loss/truediv_grad/ReshapeReshape:training/Adam/gradients/loss/weather_loss/truediv_grad/Sum<training/Adam/gradients/loss/weather_loss/truediv_grad/Shape*'
_output_shapes
:?????????*
T0*
Tshape0*,
_class"
 loc:@loss/weather_loss/truediv
?
:training/Adam/gradients/loss/weather_loss/truediv_grad/NegNegweather/Softmax*
T0*,
_class"
 loc:@loss/weather_loss/truediv*'
_output_shapes
:?????????
?
@training/Adam/gradients/loss/weather_loss/truediv_grad/RealDiv_1RealDiv:training/Adam/gradients/loss/weather_loss/truediv_grad/Negloss/weather_loss/Sum*'
_output_shapes
:?????????*
T0*,
_class"
 loc:@loss/weather_loss/truediv
?
@training/Adam/gradients/loss/weather_loss/truediv_grad/RealDiv_2RealDiv@training/Adam/gradients/loss/weather_loss/truediv_grad/RealDiv_1loss/weather_loss/Sum*
T0*,
_class"
 loc:@loss/weather_loss/truediv*'
_output_shapes
:?????????
?
:training/Adam/gradients/loss/weather_loss/truediv_grad/mulMulLtraining/Adam/gradients/loss/weather_loss/clip_by_value/Minimum_grad/Reshape@training/Adam/gradients/loss/weather_loss/truediv_grad/RealDiv_2*'
_output_shapes
:?????????*
T0*,
_class"
 loc:@loss/weather_loss/truediv
?
<training/Adam/gradients/loss/weather_loss/truediv_grad/Sum_1Sum:training/Adam/gradients/loss/weather_loss/truediv_grad/mulNtraining/Adam/gradients/loss/weather_loss/truediv_grad/BroadcastGradientArgs:1*
T0*,
_class"
 loc:@loss/weather_loss/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
@training/Adam/gradients/loss/weather_loss/truediv_grad/Reshape_1Reshape<training/Adam/gradients/loss/weather_loss/truediv_grad/Sum_1>training/Adam/gradients/loss/weather_loss/truediv_grad/Shape_1*
T0*
Tshape0*,
_class"
 loc:@loss/weather_loss/truediv*'
_output_shapes
:?????????
?
8training/Adam/gradients/loss/weather_loss/Sum_grad/ShapeShapeweather/Softmax*
T0*
out_type0*(
_class
loc:@loss/weather_loss/Sum*
_output_shapes
:
?
7training/Adam/gradients/loss/weather_loss/Sum_grad/SizeConst*
value	B :*(
_class
loc:@loss/weather_loss/Sum*
dtype0*
_output_shapes
: 
?
6training/Adam/gradients/loss/weather_loss/Sum_grad/addAdd'loss/weather_loss/Sum/reduction_indices7training/Adam/gradients/loss/weather_loss/Sum_grad/Size*
_output_shapes
: *
T0*(
_class
loc:@loss/weather_loss/Sum
?
6training/Adam/gradients/loss/weather_loss/Sum_grad/modFloorMod6training/Adam/gradients/loss/weather_loss/Sum_grad/add7training/Adam/gradients/loss/weather_loss/Sum_grad/Size*
T0*(
_class
loc:@loss/weather_loss/Sum*
_output_shapes
: 
?
:training/Adam/gradients/loss/weather_loss/Sum_grad/Shape_1Const*
valueB *(
_class
loc:@loss/weather_loss/Sum*
dtype0*
_output_shapes
: 
?
>training/Adam/gradients/loss/weather_loss/Sum_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *(
_class
loc:@loss/weather_loss/Sum
?
>training/Adam/gradients/loss/weather_loss/Sum_grad/range/deltaConst*
value	B :*(
_class
loc:@loss/weather_loss/Sum*
dtype0*
_output_shapes
: 
?
8training/Adam/gradients/loss/weather_loss/Sum_grad/rangeRange>training/Adam/gradients/loss/weather_loss/Sum_grad/range/start7training/Adam/gradients/loss/weather_loss/Sum_grad/Size>training/Adam/gradients/loss/weather_loss/Sum_grad/range/delta*(
_class
loc:@loss/weather_loss/Sum*
_output_shapes
:*

Tidx0
?
=training/Adam/gradients/loss/weather_loss/Sum_grad/Fill/valueConst*
value	B :*(
_class
loc:@loss/weather_loss/Sum*
dtype0*
_output_shapes
: 
?
7training/Adam/gradients/loss/weather_loss/Sum_grad/FillFill:training/Adam/gradients/loss/weather_loss/Sum_grad/Shape_1=training/Adam/gradients/loss/weather_loss/Sum_grad/Fill/value*
T0*

index_type0*(
_class
loc:@loss/weather_loss/Sum*
_output_shapes
: 
?
@training/Adam/gradients/loss/weather_loss/Sum_grad/DynamicStitchDynamicStitch8training/Adam/gradients/loss/weather_loss/Sum_grad/range6training/Adam/gradients/loss/weather_loss/Sum_grad/mod8training/Adam/gradients/loss/weather_loss/Sum_grad/Shape7training/Adam/gradients/loss/weather_loss/Sum_grad/Fill*
N*#
_output_shapes
:?????????*
T0*(
_class
loc:@loss/weather_loss/Sum
?
<training/Adam/gradients/loss/weather_loss/Sum_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*(
_class
loc:@loss/weather_loss/Sum
?
:training/Adam/gradients/loss/weather_loss/Sum_grad/MaximumMaximum@training/Adam/gradients/loss/weather_loss/Sum_grad/DynamicStitch<training/Adam/gradients/loss/weather_loss/Sum_grad/Maximum/y*
T0*(
_class
loc:@loss/weather_loss/Sum*#
_output_shapes
:?????????
?
;training/Adam/gradients/loss/weather_loss/Sum_grad/floordivFloorDiv8training/Adam/gradients/loss/weather_loss/Sum_grad/Shape:training/Adam/gradients/loss/weather_loss/Sum_grad/Maximum*
_output_shapes
:*
T0*(
_class
loc:@loss/weather_loss/Sum
?
:training/Adam/gradients/loss/weather_loss/Sum_grad/ReshapeReshape@training/Adam/gradients/loss/weather_loss/truediv_grad/Reshape_1@training/Adam/gradients/loss/weather_loss/Sum_grad/DynamicStitch*
T0*
Tshape0*(
_class
loc:@loss/weather_loss/Sum*
_output_shapes
:
?
7training/Adam/gradients/loss/weather_loss/Sum_grad/TileTile:training/Adam/gradients/loss/weather_loss/Sum_grad/Reshape;training/Adam/gradients/loss/weather_loss/Sum_grad/floordiv*'
_output_shapes
:?????????*

Tmultiples0*
T0*(
_class
loc:@loss/weather_loss/Sum
?
training/Adam/gradients/AddNAddNItraining/Adam/gradients/loss/ground_loss/logistic_loss/Select_grad/SelectGtraining/Adam/gradients/loss/ground_loss/logistic_loss/mul_grad/ReshapeMtraining/Adam/gradients/loss/ground_loss/logistic_loss/Select_1_grad/Select_1Ctraining/Adam/gradients/loss/ground_loss/logistic_loss/Neg_grad/Neg*
T0*8
_class.
,*loc:@loss/ground_loss/logistic_loss/Select*
N*'
_output_shapes
:?????????
?
<training/Adam/gradients/loss/ground_loss/Log_grad/Reciprocal
Reciprocalloss/ground_loss/truediv^training/Adam/gradients/AddN*
T0*'
_class
loc:@loss/ground_loss/Log*'
_output_shapes
:?????????
?
5training/Adam/gradients/loss/ground_loss/Log_grad/mulMultraining/Adam/gradients/AddN<training/Adam/gradients/loss/ground_loss/Log_grad/Reciprocal*
T0*'
_class
loc:@loss/ground_loss/Log*'
_output_shapes
:?????????
?
training/Adam/gradients/AddN_1AddN>training/Adam/gradients/loss/weather_loss/truediv_grad/Reshape7training/Adam/gradients/loss/weather_loss/Sum_grad/Tile*
T0*,
_class"
 loc:@loss/weather_loss/truediv*
N*'
_output_shapes
:?????????
?
0training/Adam/gradients/weather/Softmax_grad/mulMultraining/Adam/gradients/AddN_1weather/Softmax*'
_output_shapes
:?????????*
T0*"
_class
loc:@weather/Softmax
?
Btraining/Adam/gradients/weather/Softmax_grad/Sum/reduction_indicesConst*
valueB:*"
_class
loc:@weather/Softmax*
dtype0*
_output_shapes
:
?
0training/Adam/gradients/weather/Softmax_grad/SumSum0training/Adam/gradients/weather/Softmax_grad/mulBtraining/Adam/gradients/weather/Softmax_grad/Sum/reduction_indices*
T0*"
_class
loc:@weather/Softmax*#
_output_shapes
:?????????*

Tidx0*
	keep_dims( 
?
:training/Adam/gradients/weather/Softmax_grad/Reshape/shapeConst*
valueB"????   *"
_class
loc:@weather/Softmax*
dtype0*
_output_shapes
:
?
4training/Adam/gradients/weather/Softmax_grad/ReshapeReshape0training/Adam/gradients/weather/Softmax_grad/Sum:training/Adam/gradients/weather/Softmax_grad/Reshape/shape*
T0*
Tshape0*"
_class
loc:@weather/Softmax*'
_output_shapes
:?????????
?
0training/Adam/gradients/weather/Softmax_grad/subSubtraining/Adam/gradients/AddN_14training/Adam/gradients/weather/Softmax_grad/Reshape*'
_output_shapes
:?????????*
T0*"
_class
loc:@weather/Softmax
?
2training/Adam/gradients/weather/Softmax_grad/mul_1Mul0training/Adam/gradients/weather/Softmax_grad/subweather/Softmax*
T0*"
_class
loc:@weather/Softmax*'
_output_shapes
:?????????
?
;training/Adam/gradients/loss/ground_loss/truediv_grad/ShapeShapeloss/ground_loss/clip_by_value*
T0*
out_type0*+
_class!
loc:@loss/ground_loss/truediv*
_output_shapes
:
?
=training/Adam/gradients/loss/ground_loss/truediv_grad/Shape_1Shapeloss/ground_loss/sub_1*
_output_shapes
:*
T0*
out_type0*+
_class!
loc:@loss/ground_loss/truediv
?
Ktraining/Adam/gradients/loss/ground_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs;training/Adam/gradients/loss/ground_loss/truediv_grad/Shape=training/Adam/gradients/loss/ground_loss/truediv_grad/Shape_1*
T0*+
_class!
loc:@loss/ground_loss/truediv*2
_output_shapes 
:?????????:?????????
?
=training/Adam/gradients/loss/ground_loss/truediv_grad/RealDivRealDiv5training/Adam/gradients/loss/ground_loss/Log_grad/mulloss/ground_loss/sub_1*
T0*+
_class!
loc:@loss/ground_loss/truediv*'
_output_shapes
:?????????
?
9training/Adam/gradients/loss/ground_loss/truediv_grad/SumSum=training/Adam/gradients/loss/ground_loss/truediv_grad/RealDivKtraining/Adam/gradients/loss/ground_loss/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@loss/ground_loss/truediv
?
=training/Adam/gradients/loss/ground_loss/truediv_grad/ReshapeReshape9training/Adam/gradients/loss/ground_loss/truediv_grad/Sum;training/Adam/gradients/loss/ground_loss/truediv_grad/Shape*
T0*
Tshape0*+
_class!
loc:@loss/ground_loss/truediv*'
_output_shapes
:?????????
?
9training/Adam/gradients/loss/ground_loss/truediv_grad/NegNegloss/ground_loss/clip_by_value*
T0*+
_class!
loc:@loss/ground_loss/truediv*'
_output_shapes
:?????????
?
?training/Adam/gradients/loss/ground_loss/truediv_grad/RealDiv_1RealDiv9training/Adam/gradients/loss/ground_loss/truediv_grad/Negloss/ground_loss/sub_1*
T0*+
_class!
loc:@loss/ground_loss/truediv*'
_output_shapes
:?????????
?
?training/Adam/gradients/loss/ground_loss/truediv_grad/RealDiv_2RealDiv?training/Adam/gradients/loss/ground_loss/truediv_grad/RealDiv_1loss/ground_loss/sub_1*'
_output_shapes
:?????????*
T0*+
_class!
loc:@loss/ground_loss/truediv
?
9training/Adam/gradients/loss/ground_loss/truediv_grad/mulMul5training/Adam/gradients/loss/ground_loss/Log_grad/mul?training/Adam/gradients/loss/ground_loss/truediv_grad/RealDiv_2*
T0*+
_class!
loc:@loss/ground_loss/truediv*'
_output_shapes
:?????????
?
;training/Adam/gradients/loss/ground_loss/truediv_grad/Sum_1Sum9training/Adam/gradients/loss/ground_loss/truediv_grad/mulMtraining/Adam/gradients/loss/ground_loss/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@loss/ground_loss/truediv
?
?training/Adam/gradients/loss/ground_loss/truediv_grad/Reshape_1Reshape;training/Adam/gradients/loss/ground_loss/truediv_grad/Sum_1=training/Adam/gradients/loss/ground_loss/truediv_grad/Shape_1*
T0*
Tshape0*+
_class!
loc:@loss/ground_loss/truediv*'
_output_shapes
:?????????
?
8training/Adam/gradients/weather/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/weather/Softmax_grad/mul_1*
T0*"
_class
loc:@weather/BiasAdd*
data_formatNHWC*
_output_shapes
:
?
9training/Adam/gradients/loss/ground_loss/sub_1_grad/ShapeConst*
valueB *)
_class
loc:@loss/ground_loss/sub_1*
dtype0*
_output_shapes
: 
?
;training/Adam/gradients/loss/ground_loss/sub_1_grad/Shape_1Shapeloss/ground_loss/clip_by_value*
T0*
out_type0*)
_class
loc:@loss/ground_loss/sub_1*
_output_shapes
:
?
Itraining/Adam/gradients/loss/ground_loss/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs9training/Adam/gradients/loss/ground_loss/sub_1_grad/Shape;training/Adam/gradients/loss/ground_loss/sub_1_grad/Shape_1*
T0*)
_class
loc:@loss/ground_loss/sub_1*2
_output_shapes 
:?????????:?????????
?
7training/Adam/gradients/loss/ground_loss/sub_1_grad/SumSum?training/Adam/gradients/loss/ground_loss/truediv_grad/Reshape_1Itraining/Adam/gradients/loss/ground_loss/sub_1_grad/BroadcastGradientArgs*
T0*)
_class
loc:@loss/ground_loss/sub_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
;training/Adam/gradients/loss/ground_loss/sub_1_grad/ReshapeReshape7training/Adam/gradients/loss/ground_loss/sub_1_grad/Sum9training/Adam/gradients/loss/ground_loss/sub_1_grad/Shape*
T0*
Tshape0*)
_class
loc:@loss/ground_loss/sub_1*
_output_shapes
: 
?
9training/Adam/gradients/loss/ground_loss/sub_1_grad/Sum_1Sum?training/Adam/gradients/loss/ground_loss/truediv_grad/Reshape_1Ktraining/Adam/gradients/loss/ground_loss/sub_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*)
_class
loc:@loss/ground_loss/sub_1
?
7training/Adam/gradients/loss/ground_loss/sub_1_grad/NegNeg9training/Adam/gradients/loss/ground_loss/sub_1_grad/Sum_1*
T0*)
_class
loc:@loss/ground_loss/sub_1*
_output_shapes
:
?
=training/Adam/gradients/loss/ground_loss/sub_1_grad/Reshape_1Reshape7training/Adam/gradients/loss/ground_loss/sub_1_grad/Neg;training/Adam/gradients/loss/ground_loss/sub_1_grad/Shape_1*
T0*
Tshape0*)
_class
loc:@loss/ground_loss/sub_1*'
_output_shapes
:?????????
?
2training/Adam/gradients/weather/MatMul_grad/MatMulMatMul2training/Adam/gradients/weather/Softmax_grad/mul_1weather/MatMul/ReadVariableOp*
transpose_b(*
T0*!
_class
loc:@weather/MatMul*
transpose_a( *(
_output_shapes
:??????????
?
4training/Adam/gradients/weather/MatMul_grad/MatMul_1MatMuldropout_1/Identity2training/Adam/gradients/weather/Softmax_grad/mul_1*
transpose_a(*
_output_shapes
:	?*
transpose_b( *
T0*!
_class
loc:@weather/MatMul
?
training/Adam/gradients/AddN_2AddN=training/Adam/gradients/loss/ground_loss/truediv_grad/Reshape=training/Adam/gradients/loss/ground_loss/sub_1_grad/Reshape_1*
T0*+
_class!
loc:@loss/ground_loss/truediv*
N*'
_output_shapes
:?????????
?
Atraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/ShapeShape&loss/ground_loss/clip_by_value/Minimum*
_output_shapes
:*
T0*
out_type0*1
_class'
%#loc:@loss/ground_loss/clip_by_value
?
Ctraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/Shape_1Const*
valueB *1
_class'
%#loc:@loss/ground_loss/clip_by_value*
dtype0*
_output_shapes
: 
?
Ctraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/Shape_2Shapetraining/Adam/gradients/AddN_2*
T0*
out_type0*1
_class'
%#loc:@loss/ground_loss/clip_by_value*
_output_shapes
:
?
Gtraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/zeros/ConstConst*
valueB
 *    *1
_class'
%#loc:@loss/ground_loss/clip_by_value*
dtype0*
_output_shapes
: 
?
Atraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/zerosFillCtraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/Shape_2Gtraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/zeros/Const*
T0*

index_type0*1
_class'
%#loc:@loss/ground_loss/clip_by_value*'
_output_shapes
:?????????
?
Htraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/GreaterEqualGreaterEqual&loss/ground_loss/clip_by_value/Minimumloss/ground_loss/Const*'
_output_shapes
:?????????*
T0*1
_class'
%#loc:@loss/ground_loss/clip_by_value
?
Qtraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsAtraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/ShapeCtraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0*1
_class'
%#loc:@loss/ground_loss/clip_by_value
?
Btraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/SelectSelectHtraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/GreaterEqualtraining/Adam/gradients/AddN_2Atraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/zeros*
T0*1
_class'
%#loc:@loss/ground_loss/clip_by_value*'
_output_shapes
:?????????
?
Dtraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/Select_1SelectHtraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/GreaterEqualAtraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/zerostraining/Adam/gradients/AddN_2*
T0*1
_class'
%#loc:@loss/ground_loss/clip_by_value*'
_output_shapes
:?????????
?
?training/Adam/gradients/loss/ground_loss/clip_by_value_grad/SumSumBtraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/SelectQtraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*1
_class'
%#loc:@loss/ground_loss/clip_by_value
?
Ctraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/ReshapeReshape?training/Adam/gradients/loss/ground_loss/clip_by_value_grad/SumAtraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/Shape*'
_output_shapes
:?????????*
T0*
Tshape0*1
_class'
%#loc:@loss/ground_loss/clip_by_value
?
Atraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/Sum_1SumDtraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/Select_1Straining/Adam/gradients/loss/ground_loss/clip_by_value_grad/BroadcastGradientArgs:1*
T0*1
_class'
%#loc:@loss/ground_loss/clip_by_value*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
Etraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/Reshape_1ReshapeAtraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/Sum_1Ctraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/Shape_1*
T0*
Tshape0*1
_class'
%#loc:@loss/ground_loss/clip_by_value*
_output_shapes
: 
?
Itraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/ShapeShapeground/Sigmoid*
T0*
out_type0*9
_class/
-+loc:@loss/ground_loss/clip_by_value/Minimum*
_output_shapes
:
?
Ktraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *9
_class/
-+loc:@loss/ground_loss/clip_by_value/Minimum
?
Ktraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/Shape_2ShapeCtraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/Reshape*
_output_shapes
:*
T0*
out_type0*9
_class/
-+loc:@loss/ground_loss/clip_by_value/Minimum
?
Otraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *9
_class/
-+loc:@loss/ground_loss/clip_by_value/Minimum
?
Itraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/zerosFillKtraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/Shape_2Otraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/zeros/Const*
T0*

index_type0*9
_class/
-+loc:@loss/ground_loss/clip_by_value/Minimum*'
_output_shapes
:?????????
?
Mtraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/LessEqual	LessEqualground/Sigmoidloss/ground_loss/sub*
T0*9
_class/
-+loc:@loss/ground_loss/clip_by_value/Minimum*'
_output_shapes
:?????????
?
Ytraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsItraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/ShapeKtraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/Shape_1*
T0*9
_class/
-+loc:@loss/ground_loss/clip_by_value/Minimum*2
_output_shapes 
:?????????:?????????
?
Jtraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/SelectSelectMtraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/LessEqualCtraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/ReshapeItraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/zeros*
T0*9
_class/
-+loc:@loss/ground_loss/clip_by_value/Minimum*'
_output_shapes
:?????????
?
Ltraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/Select_1SelectMtraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/LessEqualItraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/zerosCtraining/Adam/gradients/loss/ground_loss/clip_by_value_grad/Reshape*
T0*9
_class/
-+loc:@loss/ground_loss/clip_by_value/Minimum*'
_output_shapes
:?????????
?
Gtraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/SumSumJtraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/SelectYtraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*
T0*9
_class/
-+loc:@loss/ground_loss/clip_by_value/Minimum*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
Ktraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/ReshapeReshapeGtraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/SumItraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/Shape*'
_output_shapes
:?????????*
T0*
Tshape0*9
_class/
-+loc:@loss/ground_loss/clip_by_value/Minimum
?
Itraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/Sum_1SumLtraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/Select_1[training/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*9
_class/
-+loc:@loss/ground_loss/clip_by_value/Minimum
?
Mtraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/Reshape_1ReshapeItraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/Sum_1Ktraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/Shape_1*
T0*
Tshape0*9
_class/
-+loc:@loss/ground_loss/clip_by_value/Minimum*
_output_shapes
: 
?
7training/Adam/gradients/ground/Sigmoid_grad/SigmoidGradSigmoidGradground/SigmoidKtraining/Adam/gradients/loss/ground_loss/clip_by_value/Minimum_grad/Reshape*
T0*!
_class
loc:@ground/Sigmoid*'
_output_shapes
:?????????
?
7training/Adam/gradients/ground/BiasAdd_grad/BiasAddGradBiasAddGrad7training/Adam/gradients/ground/Sigmoid_grad/SigmoidGrad*
T0*!
_class
loc:@ground/BiasAdd*
data_formatNHWC*
_output_shapes
:
?
1training/Adam/gradients/ground/MatMul_grad/MatMulMatMul7training/Adam/gradients/ground/Sigmoid_grad/SigmoidGradground/MatMul/ReadVariableOp*
T0* 
_class
loc:@ground/MatMul*
transpose_a( *(
_output_shapes
:??????????*
transpose_b(
?
3training/Adam/gradients/ground/MatMul_grad/MatMul_1MatMuldropout_1/Identity7training/Adam/gradients/ground/Sigmoid_grad/SigmoidGrad*
transpose_b( *
T0* 
_class
loc:@ground/MatMul*
transpose_a(*
_output_shapes
:	?
?
training/Adam/gradients/AddN_3AddN2training/Adam/gradients/weather/MatMul_grad/MatMul1training/Adam/gradients/ground/MatMul_grad/MatMul*
T0*!
_class
loc:@weather/MatMul*
N*(
_output_shapes
:??????????
?
2training/Adam/gradients/dense_1/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_3dense_1/Relu*
T0*
_class
loc:@dense_1/Relu*(
_output_shapes
:??????????
?
8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_1/Relu_grad/ReluGrad*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC*
_output_shapes	
:?*
T0
?
2training/Adam/gradients/dense_1/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_1/Relu_grad/ReluGraddense_1/MatMul/ReadVariableOp*
T0*!
_class
loc:@dense_1/MatMul*
transpose_a( *(
_output_shapes
:??????????*
transpose_b(
?
4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1MatMuldropout/Identity2training/Adam/gradients/dense_1/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_1/MatMul*
transpose_a(* 
_output_shapes
:
??*
transpose_b( 
?
0training/Adam/gradients/dense/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_1/MatMul_grad/MatMul
dense/Relu*
T0*
_class
loc:@dense/Relu*(
_output_shapes
:??????????
?
6training/Adam/gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad0training/Adam/gradients/dense/Relu_grad/ReluGrad*
T0* 
_class
loc:@dense/BiasAdd*
data_formatNHWC*
_output_shapes	
:?
?
0training/Adam/gradients/dense/MatMul_grad/MatMulMatMul0training/Adam/gradients/dense/Relu_grad/ReluGraddense/MatMul/ReadVariableOp*
T0*
_class
loc:@dense/MatMul*
transpose_a( *)
_output_shapes
:???????????*
transpose_b(
?
2training/Adam/gradients/dense/MatMul_grad/MatMul_1MatMulflatten/Reshape0training/Adam/gradients/dense/Relu_grad/ReluGrad*
transpose_b( *
T0*
_class
loc:@dense/MatMul*
transpose_a(*!
_output_shapes
:???
?
2training/Adam/gradients/flatten/Reshape_grad/ShapeShapemax_pooling2d_1/MaxPool*
_output_shapes
:*
T0*
out_type0*"
_class
loc:@flatten/Reshape
?
4training/Adam/gradients/flatten/Reshape_grad/ReshapeReshape0training/Adam/gradients/dense/MatMul_grad/MatMul2training/Adam/gradients/flatten/Reshape_grad/Shape*
T0*
Tshape0*"
_class
loc:@flatten/Reshape*/
_output_shapes
:?????????   
?
@training/Adam/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d_1/Relumax_pooling2d_1/MaxPool4training/Adam/gradients/flatten/Reshape_grad/Reshape*
T0**
_class 
loc:@max_pooling2d_1/MaxPool*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:?????????@@ 
?
3training/Adam/gradients/conv2d_1/Relu_grad/ReluGradReluGrad@training/Adam/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradconv2d_1/Relu*
T0* 
_class
loc:@conv2d_1/Relu*/
_output_shapes
:?????????@@ 
?
9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_1/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_1/BiasAdd*
data_formatNHWC*
_output_shapes
: 
?
3training/Adam/gradients/conv2d_1/Conv2D_grad/ShapeNShapeNmax_pooling2d/MaxPoolconv2d_1/Conv2D/ReadVariableOp*
T0*
out_type0*"
_class
loc:@conv2d_1/Conv2D*
N* 
_output_shapes
::
?
@training/Adam/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput3training/Adam/gradients/conv2d_1/Conv2D_grad/ShapeNconv2d_1/Conv2D/ReadVariableOp3training/Adam/gradients/conv2d_1/Relu_grad/ReluGrad*
paddingSAME*/
_output_shapes
:?????????@@ *
	dilations
*
T0*"
_class
loc:@conv2d_1/Conv2D*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
?
Atraining/Adam/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d/MaxPool5training/Adam/gradients/conv2d_1/Conv2D_grad/ShapeN:13training/Adam/gradients/conv2d_1/Relu_grad/ReluGrad*&
_output_shapes
:  *
	dilations
*
T0*"
_class
loc:@conv2d_1/Conv2D*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
?
>training/Adam/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d/Relumax_pooling2d/MaxPool@training/Adam/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInput*
T0*(
_class
loc:@max_pooling2d/MaxPool*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*1
_output_shapes
:??????????? 
?
1training/Adam/gradients/conv2d/Relu_grad/ReluGradReluGrad>training/Adam/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradconv2d/Relu*
T0*
_class
loc:@conv2d/Relu*1
_output_shapes
:??????????? 
?
7training/Adam/gradients/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad1training/Adam/gradients/conv2d/Relu_grad/ReluGrad*
T0*!
_class
loc:@conv2d/BiasAdd*
data_formatNHWC*
_output_shapes
: 
?
1training/Adam/gradients/conv2d/Conv2D_grad/ShapeNShapeNinput_layerconv2d/Conv2D/ReadVariableOp*
T0*
out_type0* 
_class
loc:@conv2d/Conv2D*
N* 
_output_shapes
::
?
>training/Adam/gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput1training/Adam/gradients/conv2d/Conv2D_grad/ShapeNconv2d/Conv2D/ReadVariableOp1training/Adam/gradients/conv2d/Relu_grad/ReluGrad*
paddingSAME*1
_output_shapes
:???????????*
	dilations
*
T0* 
_class
loc:@conv2d/Conv2D*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
?
?training/Adam/gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterinput_layer3training/Adam/gradients/conv2d/Conv2D_grad/ShapeN:11training/Adam/gradients/conv2d/Relu_grad/ReluGrad*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0* 
_class
loc:@conv2d/Conv2D*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
U
training/Adam/ConstConst*
value	B	 R*
dtype0	*
_output_shapes
: 
k
!training/Adam/AssignAddVariableOpAssignAddVariableOpAdam/iterationstraining/Adam/Const*
dtype0	
?
training/Adam/ReadVariableOpReadVariableOpAdam/iterations"^training/Adam/AssignAddVariableOp*
dtype0	*
_output_shapes
: 
i
!training/Adam/Cast/ReadVariableOpReadVariableOpAdam/iterations*
dtype0	*
_output_shapes
: 
m
training/Adam/CastCast!training/Adam/Cast/ReadVariableOp*

SrcT0	*

DstT0*
_output_shapes
: 
X
training/Adam/add/yConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
b
training/Adam/addAddtraining/Adam/Casttraining/Adam/add/y*
T0*
_output_shapes
: 
d
 training/Adam/Pow/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
n
training/Adam/PowPow training/Adam/Pow/ReadVariableOptraining/Adam/add*
T0*
_output_shapes
: 
X
training/Adam/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
T0*
_output_shapes
: 
Z
training/Adam/Const_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_2*
_output_shapes
: *
T0
?
training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const_1*
T0*
_output_shapes
: 
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
_output_shapes
: *
T0
f
"training/Adam/Pow_1/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
r
training/Adam/Pow_1Pow"training/Adam/Pow_1/ReadVariableOptraining/Adam/add*
T0*
_output_shapes
: 
Z
training/Adam/sub_1/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
^
training/Adam/ReadVariableOp_1ReadVariableOpAdam/lr*
dtype0*
_output_shapes
: 
p
training/Adam/mulMultraining/Adam/ReadVariableOp_1training/Adam/truediv*
T0*
_output_shapes
: 
x
training/Adam/zerosConst*%
valueB *    *
dtype0*&
_output_shapes
: 
?
training/Adam/VariableVarHandleOp*
dtype0*
	container *
_output_shapes
: *
shape: *'
shared_nametraining/Adam/Variable
}
7training/Adam/Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable*
_output_shapes
: 
?
training/Adam/Variable/AssignAssignVariableOptraining/Adam/Variabletraining/Adam/zeros*)
_class
loc:@training/Adam/Variable*
dtype0
?
*training/Adam/Variable/Read/ReadVariableOpReadVariableOptraining/Adam/Variable*
dtype0*&
_output_shapes
: *)
_class
loc:@training/Adam/Variable
b
training/Adam/zeros_1Const*
valueB *    *
dtype0*
_output_shapes
: 
?
training/Adam/Variable_1VarHandleOp*
dtype0*
	container *
_output_shapes
: *
shape: *)
shared_nametraining/Adam/Variable_1
?
9training/Adam/Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_1*
_output_shapes
: 
?
training/Adam/Variable_1/AssignAssignVariableOptraining/Adam/Variable_1training/Adam/zeros_1*
dtype0*+
_class!
loc:@training/Adam/Variable_1
?
,training/Adam/Variable_1/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0*
_output_shapes
: 
~
%training/Adam/zeros_2/shape_as_tensorConst*%
valueB"              *
dtype0*
_output_shapes
:
`
training/Adam/zeros_2/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
?
training/Adam/zeros_2Fill%training/Adam/zeros_2/shape_as_tensortraining/Adam/zeros_2/Const*
T0*

index_type0*&
_output_shapes
:  
?
training/Adam/Variable_2VarHandleOp*)
shared_nametraining/Adam/Variable_2*
dtype0*
	container *
_output_shapes
: *
shape:  
?
9training/Adam/Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_2*
_output_shapes
: 
?
training/Adam/Variable_2/AssignAssignVariableOptraining/Adam/Variable_2training/Adam/zeros_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0
?
,training/Adam/Variable_2/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0*&
_output_shapes
:  
b
training/Adam/zeros_3Const*
valueB *    *
dtype0*
_output_shapes
: 
?
training/Adam/Variable_3VarHandleOp*
shape: *)
shared_nametraining/Adam/Variable_3*
dtype0*
	container *
_output_shapes
: 
?
9training/Adam/Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_3*
_output_shapes
: 
?
training/Adam/Variable_3/AssignAssignVariableOptraining/Adam/Variable_3training/Adam/zeros_3*+
_class!
loc:@training/Adam/Variable_3*
dtype0
?
,training/Adam/Variable_3/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_3*+
_class!
loc:@training/Adam/Variable_3*
dtype0*
_output_shapes
: 
v
%training/Adam/zeros_4/shape_as_tensorConst*
valueB" ?  ?   *
dtype0*
_output_shapes
:
`
training/Adam/zeros_4/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
training/Adam/zeros_4Fill%training/Adam/zeros_4/shape_as_tensortraining/Adam/zeros_4/Const*
T0*

index_type0*!
_output_shapes
:???
?
training/Adam/Variable_4VarHandleOp*
dtype0*
	container *
_output_shapes
: *
shape:???*)
shared_nametraining/Adam/Variable_4
?
9training/Adam/Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_4*
_output_shapes
: 
?
training/Adam/Variable_4/AssignAssignVariableOptraining/Adam/Variable_4training/Adam/zeros_4*+
_class!
loc:@training/Adam/Variable_4*
dtype0
?
,training/Adam/Variable_4/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_4*+
_class!
loc:@training/Adam/Variable_4*
dtype0*!
_output_shapes
:???
d
training/Adam/zeros_5Const*
valueB?*    *
dtype0*
_output_shapes	
:?
?
training/Adam/Variable_5VarHandleOp*)
shared_nametraining/Adam/Variable_5*
dtype0*
	container *
_output_shapes
: *
shape:?
?
9training/Adam/Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_5*
_output_shapes
: 
?
training/Adam/Variable_5/AssignAssignVariableOptraining/Adam/Variable_5training/Adam/zeros_5*
dtype0*+
_class!
loc:@training/Adam/Variable_5
?
,training/Adam/Variable_5/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_5*
dtype0*
_output_shapes	
:?*+
_class!
loc:@training/Adam/Variable_5
v
%training/Adam/zeros_6/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"?   ?   
`
training/Adam/zeros_6/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
training/Adam/zeros_6Fill%training/Adam/zeros_6/shape_as_tensortraining/Adam/zeros_6/Const* 
_output_shapes
:
??*
T0*

index_type0
?
training/Adam/Variable_6VarHandleOp*
shape:
??*)
shared_nametraining/Adam/Variable_6*
dtype0*
	container *
_output_shapes
: 
?
9training/Adam/Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_6*
_output_shapes
: 
?
training/Adam/Variable_6/AssignAssignVariableOptraining/Adam/Variable_6training/Adam/zeros_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0
?
,training/Adam/Variable_6/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0* 
_output_shapes
:
??
d
training/Adam/zeros_7Const*
dtype0*
_output_shapes	
:?*
valueB?*    
?
training/Adam/Variable_7VarHandleOp*
shape:?*)
shared_nametraining/Adam/Variable_7*
dtype0*
	container *
_output_shapes
: 
?
9training/Adam/Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_7*
_output_shapes
: 
?
training/Adam/Variable_7/AssignAssignVariableOptraining/Adam/Variable_7training/Adam/zeros_7*+
_class!
loc:@training/Adam/Variable_7*
dtype0
?
,training/Adam/Variable_7/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_7*
dtype0*
_output_shapes	
:?*+
_class!
loc:@training/Adam/Variable_7
l
training/Adam/zeros_8Const*
valueB	?*    *
dtype0*
_output_shapes
:	?
?
training/Adam/Variable_8VarHandleOp*
dtype0*
	container *
_output_shapes
: *
shape:	?*)
shared_nametraining/Adam/Variable_8
?
9training/Adam/Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_8*
_output_shapes
: 
?
training/Adam/Variable_8/AssignAssignVariableOptraining/Adam/Variable_8training/Adam/zeros_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0
?
,training/Adam/Variable_8/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0*
_output_shapes
:	?
b
training/Adam/zeros_9Const*
valueB*    *
dtype0*
_output_shapes
:
?
training/Adam/Variable_9VarHandleOp*
dtype0*
	container *
_output_shapes
: *
shape:*)
shared_nametraining/Adam/Variable_9
?
9training/Adam/Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_9*
_output_shapes
: 
?
training/Adam/Variable_9/AssignAssignVariableOptraining/Adam/Variable_9training/Adam/zeros_9*+
_class!
loc:@training/Adam/Variable_9*
dtype0
?
,training/Adam/Variable_9/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9*
dtype0*
_output_shapes
:
w
&training/Adam/zeros_10/shape_as_tensorConst*
valueB"?      *
dtype0*
_output_shapes
:
a
training/Adam/zeros_10/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
?
training/Adam/zeros_10Fill&training/Adam/zeros_10/shape_as_tensortraining/Adam/zeros_10/Const*
T0*

index_type0*
_output_shapes
:	?
?
training/Adam/Variable_10VarHandleOp*
dtype0*
	container *
_output_shapes
: *
shape:	?**
shared_nametraining/Adam/Variable_10
?
:training/Adam/Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_10*
_output_shapes
: 
?
 training/Adam/Variable_10/AssignAssignVariableOptraining/Adam/Variable_10training/Adam/zeros_10*,
_class"
 loc:@training/Adam/Variable_10*
dtype0
?
-training/Adam/Variable_10/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_10*,
_class"
 loc:@training/Adam/Variable_10*
dtype0*
_output_shapes
:	?
c
training/Adam/zeros_11Const*
valueB*    *
dtype0*
_output_shapes
:
?
training/Adam/Variable_11VarHandleOp*
dtype0*
	container *
_output_shapes
: *
shape:**
shared_nametraining/Adam/Variable_11
?
:training/Adam/Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_11*
_output_shapes
: 
?
 training/Adam/Variable_11/AssignAssignVariableOptraining/Adam/Variable_11training/Adam/zeros_11*
dtype0*,
_class"
 loc:@training/Adam/Variable_11
?
-training/Adam/Variable_11/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0*
_output_shapes
:
{
training/Adam/zeros_12Const*%
valueB *    *
dtype0*&
_output_shapes
: 
?
training/Adam/Variable_12VarHandleOp*
dtype0*
	container *
_output_shapes
: *
shape: **
shared_nametraining/Adam/Variable_12
?
:training/Adam/Variable_12/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_12*
_output_shapes
: 
?
 training/Adam/Variable_12/AssignAssignVariableOptraining/Adam/Variable_12training/Adam/zeros_12*,
_class"
 loc:@training/Adam/Variable_12*
dtype0
?
-training/Adam/Variable_12/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_12*
dtype0*&
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_12
c
training/Adam/zeros_13Const*
valueB *    *
dtype0*
_output_shapes
: 
?
training/Adam/Variable_13VarHandleOp**
shared_nametraining/Adam/Variable_13*
dtype0*
	container *
_output_shapes
: *
shape: 
?
:training/Adam/Variable_13/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_13*
_output_shapes
: 
?
 training/Adam/Variable_13/AssignAssignVariableOptraining/Adam/Variable_13training/Adam/zeros_13*,
_class"
 loc:@training/Adam/Variable_13*
dtype0
?
-training/Adam/Variable_13/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_13*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_13

&training/Adam/zeros_14/shape_as_tensorConst*%
valueB"              *
dtype0*
_output_shapes
:
a
training/Adam/zeros_14/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
training/Adam/zeros_14Fill&training/Adam/zeros_14/shape_as_tensortraining/Adam/zeros_14/Const*
T0*

index_type0*&
_output_shapes
:  
?
training/Adam/Variable_14VarHandleOp**
shared_nametraining/Adam/Variable_14*
dtype0*
	container *
_output_shapes
: *
shape:  
?
:training/Adam/Variable_14/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_14*
_output_shapes
: 
?
 training/Adam/Variable_14/AssignAssignVariableOptraining/Adam/Variable_14training/Adam/zeros_14*
dtype0*,
_class"
 loc:@training/Adam/Variable_14
?
-training/Adam/Variable_14/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
dtype0*&
_output_shapes
:  
c
training/Adam/zeros_15Const*
dtype0*
_output_shapes
: *
valueB *    
?
training/Adam/Variable_15VarHandleOp*
dtype0*
	container *
_output_shapes
: *
shape: **
shared_nametraining/Adam/Variable_15
?
:training/Adam/Variable_15/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_15*
_output_shapes
: 
?
 training/Adam/Variable_15/AssignAssignVariableOptraining/Adam/Variable_15training/Adam/zeros_15*,
_class"
 loc:@training/Adam/Variable_15*
dtype0
?
-training/Adam/Variable_15/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_15*,
_class"
 loc:@training/Adam/Variable_15*
dtype0*
_output_shapes
: 
w
&training/Adam/zeros_16/shape_as_tensorConst*
valueB" ?  ?   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
training/Adam/zeros_16Fill&training/Adam/zeros_16/shape_as_tensortraining/Adam/zeros_16/Const*!
_output_shapes
:???*
T0*

index_type0
?
training/Adam/Variable_16VarHandleOp*
dtype0*
	container *
_output_shapes
: *
shape:???**
shared_nametraining/Adam/Variable_16
?
:training/Adam/Variable_16/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_16*
_output_shapes
: 
?
 training/Adam/Variable_16/AssignAssignVariableOptraining/Adam/Variable_16training/Adam/zeros_16*,
_class"
 loc:@training/Adam/Variable_16*
dtype0
?
-training/Adam/Variable_16/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
dtype0*!
_output_shapes
:???
e
training/Adam/zeros_17Const*
valueB?*    *
dtype0*
_output_shapes	
:?
?
training/Adam/Variable_17VarHandleOp*
dtype0*
	container *
_output_shapes
: *
shape:?**
shared_nametraining/Adam/Variable_17
?
:training/Adam/Variable_17/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_17*
_output_shapes
: 
?
 training/Adam/Variable_17/AssignAssignVariableOptraining/Adam/Variable_17training/Adam/zeros_17*
dtype0*,
_class"
 loc:@training/Adam/Variable_17
?
-training/Adam/Variable_17/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_17*,
_class"
 loc:@training/Adam/Variable_17*
dtype0*
_output_shapes	
:?
w
&training/Adam/zeros_18/shape_as_tensorConst*
valueB"?   ?   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_18/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
training/Adam/zeros_18Fill&training/Adam/zeros_18/shape_as_tensortraining/Adam/zeros_18/Const*
T0*

index_type0* 
_output_shapes
:
??
?
training/Adam/Variable_18VarHandleOp*
dtype0*
	container *
_output_shapes
: *
shape:
??**
shared_nametraining/Adam/Variable_18
?
:training/Adam/Variable_18/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_18*
_output_shapes
: 
?
 training/Adam/Variable_18/AssignAssignVariableOptraining/Adam/Variable_18training/Adam/zeros_18*,
_class"
 loc:@training/Adam/Variable_18*
dtype0
?
-training/Adam/Variable_18/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_18*
dtype0* 
_output_shapes
:
??*,
_class"
 loc:@training/Adam/Variable_18
e
training/Adam/zeros_19Const*
valueB?*    *
dtype0*
_output_shapes	
:?
?
training/Adam/Variable_19VarHandleOp*
dtype0*
	container *
_output_shapes
: *
shape:?**
shared_nametraining/Adam/Variable_19
?
:training/Adam/Variable_19/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_19*
_output_shapes
: 
?
 training/Adam/Variable_19/AssignAssignVariableOptraining/Adam/Variable_19training/Adam/zeros_19*,
_class"
 loc:@training/Adam/Variable_19*
dtype0
?
-training/Adam/Variable_19/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_19*,
_class"
 loc:@training/Adam/Variable_19*
dtype0*
_output_shapes	
:?
m
training/Adam/zeros_20Const*
valueB	?*    *
dtype0*
_output_shapes
:	?
?
training/Adam/Variable_20VarHandleOp*
dtype0*
	container *
_output_shapes
: *
shape:	?**
shared_nametraining/Adam/Variable_20
?
:training/Adam/Variable_20/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_20*
_output_shapes
: 
?
 training/Adam/Variable_20/AssignAssignVariableOptraining/Adam/Variable_20training/Adam/zeros_20*,
_class"
 loc:@training/Adam/Variable_20*
dtype0
?
-training/Adam/Variable_20/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_20*,
_class"
 loc:@training/Adam/Variable_20*
dtype0*
_output_shapes
:	?
c
training/Adam/zeros_21Const*
valueB*    *
dtype0*
_output_shapes
:
?
training/Adam/Variable_21VarHandleOp*
shape:**
shared_nametraining/Adam/Variable_21*
dtype0*
	container *
_output_shapes
: 
?
:training/Adam/Variable_21/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_21*
_output_shapes
: 
?
 training/Adam/Variable_21/AssignAssignVariableOptraining/Adam/Variable_21training/Adam/zeros_21*,
_class"
 loc:@training/Adam/Variable_21*
dtype0
?
-training/Adam/Variable_21/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
dtype0*
_output_shapes
:
w
&training/Adam/zeros_22/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"?      
a
training/Adam/zeros_22/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
training/Adam/zeros_22Fill&training/Adam/zeros_22/shape_as_tensortraining/Adam/zeros_22/Const*
T0*

index_type0*
_output_shapes
:	?
?
training/Adam/Variable_22VarHandleOp*
shape:	?**
shared_nametraining/Adam/Variable_22*
dtype0*
	container *
_output_shapes
: 
?
:training/Adam/Variable_22/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_22*
_output_shapes
: 
?
 training/Adam/Variable_22/AssignAssignVariableOptraining/Adam/Variable_22training/Adam/zeros_22*,
_class"
 loc:@training/Adam/Variable_22*
dtype0
?
-training/Adam/Variable_22/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_22*,
_class"
 loc:@training/Adam/Variable_22*
dtype0*
_output_shapes
:	?
c
training/Adam/zeros_23Const*
valueB*    *
dtype0*
_output_shapes
:
?
training/Adam/Variable_23VarHandleOp*
dtype0*
	container *
_output_shapes
: *
shape:**
shared_nametraining/Adam/Variable_23
?
:training/Adam/Variable_23/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_23*
_output_shapes
: 
?
 training/Adam/Variable_23/AssignAssignVariableOptraining/Adam/Variable_23training/Adam/zeros_23*,
_class"
 loc:@training/Adam/Variable_23*
dtype0
?
-training/Adam/Variable_23/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_23*,
_class"
 loc:@training/Adam/Variable_23*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_24/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_24/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
training/Adam/zeros_24Fill&training/Adam/zeros_24/shape_as_tensortraining/Adam/zeros_24/Const*
T0*

index_type0*
_output_shapes
:
?
training/Adam/Variable_24VarHandleOp*
shape:**
shared_nametraining/Adam/Variable_24*
dtype0*
	container *
_output_shapes
: 
?
:training/Adam/Variable_24/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_24*
_output_shapes
: 
?
 training/Adam/Variable_24/AssignAssignVariableOptraining/Adam/Variable_24training/Adam/zeros_24*,
_class"
 loc:@training/Adam/Variable_24*
dtype0
?
-training/Adam/Variable_24/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_24*,
_class"
 loc:@training/Adam/Variable_24*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_25/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_25/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
?
training/Adam/zeros_25Fill&training/Adam/zeros_25/shape_as_tensortraining/Adam/zeros_25/Const*
T0*

index_type0*
_output_shapes
:
?
training/Adam/Variable_25VarHandleOp**
shared_nametraining/Adam/Variable_25*
dtype0*
	container *
_output_shapes
: *
shape:
?
:training/Adam/Variable_25/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_25*
_output_shapes
: 
?
 training/Adam/Variable_25/AssignAssignVariableOptraining/Adam/Variable_25training/Adam/zeros_25*,
_class"
 loc:@training/Adam/Variable_25*
dtype0
?
-training/Adam/Variable_25/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_25*,
_class"
 loc:@training/Adam/Variable_25*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_26/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_26/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
training/Adam/zeros_26Fill&training/Adam/zeros_26/shape_as_tensortraining/Adam/zeros_26/Const*
T0*

index_type0*
_output_shapes
:
?
training/Adam/Variable_26VarHandleOp*
dtype0*
	container *
_output_shapes
: *
shape:**
shared_nametraining/Adam/Variable_26
?
:training/Adam/Variable_26/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_26*
_output_shapes
: 
?
 training/Adam/Variable_26/AssignAssignVariableOptraining/Adam/Variable_26training/Adam/zeros_26*,
_class"
 loc:@training/Adam/Variable_26*
dtype0
?
-training/Adam/Variable_26/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_26*,
_class"
 loc:@training/Adam/Variable_26*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_27/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_27/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
training/Adam/zeros_27Fill&training/Adam/zeros_27/shape_as_tensortraining/Adam/zeros_27/Const*
T0*

index_type0*
_output_shapes
:
?
training/Adam/Variable_27VarHandleOp**
shared_nametraining/Adam/Variable_27*
dtype0*
	container *
_output_shapes
: *
shape:
?
:training/Adam/Variable_27/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_27*
_output_shapes
: 
?
 training/Adam/Variable_27/AssignAssignVariableOptraining/Adam/Variable_27training/Adam/zeros_27*
dtype0*,
_class"
 loc:@training/Adam/Variable_27
?
-training/Adam/Variable_27/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_27*
dtype0*
_output_shapes
:*,
_class"
 loc:@training/Adam/Variable_27
p
&training/Adam/zeros_28/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_28/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
training/Adam/zeros_28Fill&training/Adam/zeros_28/shape_as_tensortraining/Adam/zeros_28/Const*
T0*

index_type0*
_output_shapes
:
?
training/Adam/Variable_28VarHandleOp**
shared_nametraining/Adam/Variable_28*
dtype0*
	container *
_output_shapes
: *
shape:
?
:training/Adam/Variable_28/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_28*
_output_shapes
: 
?
 training/Adam/Variable_28/AssignAssignVariableOptraining/Adam/Variable_28training/Adam/zeros_28*,
_class"
 loc:@training/Adam/Variable_28*
dtype0
?
-training/Adam/Variable_28/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_28*,
_class"
 loc:@training/Adam/Variable_28*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_29/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_29/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
training/Adam/zeros_29Fill&training/Adam/zeros_29/shape_as_tensortraining/Adam/zeros_29/Const*
T0*

index_type0*
_output_shapes
:
?
training/Adam/Variable_29VarHandleOp**
shared_nametraining/Adam/Variable_29*
dtype0*
	container *
_output_shapes
: *
shape:
?
:training/Adam/Variable_29/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_29*
_output_shapes
: 
?
 training/Adam/Variable_29/AssignAssignVariableOptraining/Adam/Variable_29training/Adam/zeros_29*,
_class"
 loc:@training/Adam/Variable_29*
dtype0
?
-training/Adam/Variable_29/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_29*,
_class"
 loc:@training/Adam/Variable_29*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_30/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_30/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
training/Adam/zeros_30Fill&training/Adam/zeros_30/shape_as_tensortraining/Adam/zeros_30/Const*
T0*

index_type0*
_output_shapes
:
?
training/Adam/Variable_30VarHandleOp*
shape:**
shared_nametraining/Adam/Variable_30*
dtype0*
	container *
_output_shapes
: 
?
:training/Adam/Variable_30/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_30*
_output_shapes
: 
?
 training/Adam/Variable_30/AssignAssignVariableOptraining/Adam/Variable_30training/Adam/zeros_30*
dtype0*,
_class"
 loc:@training/Adam/Variable_30
?
-training/Adam/Variable_30/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_30*,
_class"
 loc:@training/Adam/Variable_30*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_31/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_31/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
training/Adam/zeros_31Fill&training/Adam/zeros_31/shape_as_tensortraining/Adam/zeros_31/Const*

index_type0*
_output_shapes
:*
T0
?
training/Adam/Variable_31VarHandleOp*
shape:**
shared_nametraining/Adam/Variable_31*
dtype0*
	container *
_output_shapes
: 
?
:training/Adam/Variable_31/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_31*
_output_shapes
: 
?
 training/Adam/Variable_31/AssignAssignVariableOptraining/Adam/Variable_31training/Adam/zeros_31*,
_class"
 loc:@training/Adam/Variable_31*
dtype0
?
-training/Adam/Variable_31/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_31*,
_class"
 loc:@training/Adam/Variable_31*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_32/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_32/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
training/Adam/zeros_32Fill&training/Adam/zeros_32/shape_as_tensortraining/Adam/zeros_32/Const*
T0*

index_type0*
_output_shapes
:
?
training/Adam/Variable_32VarHandleOp*
dtype0*
	container *
_output_shapes
: *
shape:**
shared_nametraining/Adam/Variable_32
?
:training/Adam/Variable_32/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_32*
_output_shapes
: 
?
 training/Adam/Variable_32/AssignAssignVariableOptraining/Adam/Variable_32training/Adam/zeros_32*,
_class"
 loc:@training/Adam/Variable_32*
dtype0
?
-training/Adam/Variable_32/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_32*
dtype0*
_output_shapes
:*,
_class"
 loc:@training/Adam/Variable_32
p
&training/Adam/zeros_33/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_33/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
training/Adam/zeros_33Fill&training/Adam/zeros_33/shape_as_tensortraining/Adam/zeros_33/Const*
T0*

index_type0*
_output_shapes
:
?
training/Adam/Variable_33VarHandleOp*
shape:**
shared_nametraining/Adam/Variable_33*
dtype0*
	container *
_output_shapes
: 
?
:training/Adam/Variable_33/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_33*
_output_shapes
: 
?
 training/Adam/Variable_33/AssignAssignVariableOptraining/Adam/Variable_33training/Adam/zeros_33*,
_class"
 loc:@training/Adam/Variable_33*
dtype0
?
-training/Adam/Variable_33/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_33*,
_class"
 loc:@training/Adam/Variable_33*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_34/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_34/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
?
training/Adam/zeros_34Fill&training/Adam/zeros_34/shape_as_tensortraining/Adam/zeros_34/Const*
T0*

index_type0*
_output_shapes
:
?
training/Adam/Variable_34VarHandleOp*
shape:**
shared_nametraining/Adam/Variable_34*
dtype0*
	container *
_output_shapes
: 
?
:training/Adam/Variable_34/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_34*
_output_shapes
: 
?
 training/Adam/Variable_34/AssignAssignVariableOptraining/Adam/Variable_34training/Adam/zeros_34*,
_class"
 loc:@training/Adam/Variable_34*
dtype0
?
-training/Adam/Variable_34/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_34*,
_class"
 loc:@training/Adam/Variable_34*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_35/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_35/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
training/Adam/zeros_35Fill&training/Adam/zeros_35/shape_as_tensortraining/Adam/zeros_35/Const*
T0*

index_type0*
_output_shapes
:
?
training/Adam/Variable_35VarHandleOp*
shape:**
shared_nametraining/Adam/Variable_35*
dtype0*
	container *
_output_shapes
: 
?
:training/Adam/Variable_35/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_35*
_output_shapes
: 
?
 training/Adam/Variable_35/AssignAssignVariableOptraining/Adam/Variable_35training/Adam/zeros_35*,
_class"
 loc:@training/Adam/Variable_35*
dtype0
?
-training/Adam/Variable_35/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_35*
dtype0*
_output_shapes
:*,
_class"
 loc:@training/Adam/Variable_35
b
training/Adam/ReadVariableOp_2ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
?
"training/Adam/mul_1/ReadVariableOpReadVariableOptraining/Adam/Variable*
dtype0*&
_output_shapes
: 
?
training/Adam/mul_1Multraining/Adam/ReadVariableOp_2"training/Adam/mul_1/ReadVariableOp*
T0*&
_output_shapes
: 
b
training/Adam/ReadVariableOp_3ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_2/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
r
training/Adam/sub_2Subtraining/Adam/sub_2/xtraining/Adam/ReadVariableOp_3*
T0*
_output_shapes
: 
?
training/Adam/mul_2Multraining/Adam/sub_2?training/Adam/gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
u
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*&
_output_shapes
: *
T0
b
training/Adam/ReadVariableOp_4ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
?
"training/Adam/mul_3/ReadVariableOpReadVariableOptraining/Adam/Variable_12*
dtype0*&
_output_shapes
: 
?
training/Adam/mul_3Multraining/Adam/ReadVariableOp_4"training/Adam/mul_3/ReadVariableOp*
T0*&
_output_shapes
: 
b
training/Adam/ReadVariableOp_5ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_3/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
r
training/Adam/sub_3Subtraining/Adam/sub_3/xtraining/Adam/ReadVariableOp_5*
_output_shapes
: *
T0
?
training/Adam/SquareSquare?training/Adam/gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
v
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0*&
_output_shapes
: 
u
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
T0*&
_output_shapes
: 
s
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0*&
_output_shapes
: 
Z
training/Adam/Const_3Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_4Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
?
%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_4*
T0*&
_output_shapes
: 
?
training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_3*&
_output_shapes
: *
T0
l
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*&
_output_shapes
: 
Z
training/Adam/add_3/yConst*
valueB
 *???3*
dtype0*
_output_shapes
: 
x
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
T0*&
_output_shapes
: 
}
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
T0*&
_output_shapes
: 
t
training/Adam/ReadVariableOp_6ReadVariableOpconv2d/kernel*
dtype0*&
_output_shapes
: 
?
training/Adam/sub_4Subtraining/Adam/ReadVariableOp_6training/Adam/truediv_1*
T0*&
_output_shapes
: 
l
training/Adam/AssignVariableOpAssignVariableOptraining/Adam/Variabletraining/Adam/add_1*
dtype0
?
training/Adam/ReadVariableOp_7ReadVariableOptraining/Adam/Variable^training/Adam/AssignVariableOp*
dtype0*&
_output_shapes
: 
q
 training/Adam/AssignVariableOp_1AssignVariableOptraining/Adam/Variable_12training/Adam/add_2*
dtype0
?
training/Adam/ReadVariableOp_8ReadVariableOptraining/Adam/Variable_12!^training/Adam/AssignVariableOp_1*
dtype0*&
_output_shapes
: 
e
 training/Adam/AssignVariableOp_2AssignVariableOpconv2d/kerneltraining/Adam/sub_4*
dtype0
?
training/Adam/ReadVariableOp_9ReadVariableOpconv2d/kernel!^training/Adam/AssignVariableOp_2*
dtype0*&
_output_shapes
: 
c
training/Adam/ReadVariableOp_10ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
w
"training/Adam/mul_6/ReadVariableOpReadVariableOptraining/Adam/Variable_1*
dtype0*
_output_shapes
: 
?
training/Adam/mul_6Multraining/Adam/ReadVariableOp_10"training/Adam/mul_6/ReadVariableOp*
T0*
_output_shapes
: 
c
training/Adam/ReadVariableOp_11ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_5/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
s
training/Adam/sub_5Subtraining/Adam/sub_5/xtraining/Adam/ReadVariableOp_11*
T0*
_output_shapes
: 
?
training/Adam/mul_7Multraining/Adam/sub_57training/Adam/gradients/conv2d/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes
: 
c
training/Adam/ReadVariableOp_12ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
x
"training/Adam/mul_8/ReadVariableOpReadVariableOptraining/Adam/Variable_13*
dtype0*
_output_shapes
: 
?
training/Adam/mul_8Multraining/Adam/ReadVariableOp_12"training/Adam/mul_8/ReadVariableOp*
T0*
_output_shapes
: 
c
training/Adam/ReadVariableOp_13ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_6/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
s
training/Adam/sub_6Subtraining/Adam/sub_6/xtraining/Adam/ReadVariableOp_13*
T0*
_output_shapes
: 
~
training/Adam/Square_1Square7training/Adam/gradients/conv2d/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
_output_shapes
: *
T0
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
_output_shapes
: *
T0
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes
: 
Z
training/Adam/Const_5Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_6Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
?
%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_6*
T0*
_output_shapes
: 
?
training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_5*
T0*
_output_shapes
: 
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
T0*
_output_shapes
: 
Z
training/Adam/add_6/yConst*
valueB
 *???3*
dtype0*
_output_shapes
: 
l
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
_output_shapes
: *
T0
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
T0*
_output_shapes
: 
g
training/Adam/ReadVariableOp_14ReadVariableOpconv2d/bias*
dtype0*
_output_shapes
: 
y
training/Adam/sub_7Subtraining/Adam/ReadVariableOp_14training/Adam/truediv_2*
T0*
_output_shapes
: 
p
 training/Adam/AssignVariableOp_3AssignVariableOptraining/Adam/Variable_1training/Adam/add_4*
dtype0
?
training/Adam/ReadVariableOp_15ReadVariableOptraining/Adam/Variable_1!^training/Adam/AssignVariableOp_3*
dtype0*
_output_shapes
: 
q
 training/Adam/AssignVariableOp_4AssignVariableOptraining/Adam/Variable_13training/Adam/add_5*
dtype0
?
training/Adam/ReadVariableOp_16ReadVariableOptraining/Adam/Variable_13!^training/Adam/AssignVariableOp_4*
dtype0*
_output_shapes
: 
c
 training/Adam/AssignVariableOp_5AssignVariableOpconv2d/biastraining/Adam/sub_7*
dtype0
?
training/Adam/ReadVariableOp_17ReadVariableOpconv2d/bias!^training/Adam/AssignVariableOp_5*
dtype0*
_output_shapes
: 
c
training/Adam/ReadVariableOp_18ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
?
#training/Adam/mul_11/ReadVariableOpReadVariableOptraining/Adam/Variable_2*
dtype0*&
_output_shapes
:  
?
training/Adam/mul_11Multraining/Adam/ReadVariableOp_18#training/Adam/mul_11/ReadVariableOp*
T0*&
_output_shapes
:  
c
training/Adam/ReadVariableOp_19ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_8/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
s
training/Adam/sub_8Subtraining/Adam/sub_8/xtraining/Adam/ReadVariableOp_19*
_output_shapes
: *
T0
?
training/Adam/mul_12Multraining/Adam/sub_8Atraining/Adam/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:  
w
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
T0*&
_output_shapes
:  
c
training/Adam/ReadVariableOp_20ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
?
#training/Adam/mul_13/ReadVariableOpReadVariableOptraining/Adam/Variable_14*
dtype0*&
_output_shapes
:  
?
training/Adam/mul_13Multraining/Adam/ReadVariableOp_20#training/Adam/mul_13/ReadVariableOp*
T0*&
_output_shapes
:  
c
training/Adam/ReadVariableOp_21ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_9/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
s
training/Adam/sub_9Subtraining/Adam/sub_9/xtraining/Adam/ReadVariableOp_21*
T0*
_output_shapes
: 
?
training/Adam/Square_2SquareAtraining/Adam/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:  *
T0
y
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*&
_output_shapes
:  
w
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
T0*&
_output_shapes
:  
t
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
T0*&
_output_shapes
:  
Z
training/Adam/Const_7Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_8Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
?
%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_8*&
_output_shapes
:  *
T0
?
training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_7*
T0*&
_output_shapes
:  
l
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*&
_output_shapes
:  
Z
training/Adam/add_9/yConst*
valueB
 *???3*
dtype0*
_output_shapes
: 
x
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
T0*&
_output_shapes
:  
~
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
T0*&
_output_shapes
:  
w
training/Adam/ReadVariableOp_22ReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:  
?
training/Adam/sub_10Subtraining/Adam/ReadVariableOp_22training/Adam/truediv_3*
T0*&
_output_shapes
:  
p
 training/Adam/AssignVariableOp_6AssignVariableOptraining/Adam/Variable_2training/Adam/add_7*
dtype0
?
training/Adam/ReadVariableOp_23ReadVariableOptraining/Adam/Variable_2!^training/Adam/AssignVariableOp_6*
dtype0*&
_output_shapes
:  
q
 training/Adam/AssignVariableOp_7AssignVariableOptraining/Adam/Variable_14training/Adam/add_8*
dtype0
?
training/Adam/ReadVariableOp_24ReadVariableOptraining/Adam/Variable_14!^training/Adam/AssignVariableOp_7*
dtype0*&
_output_shapes
:  
h
 training/Adam/AssignVariableOp_8AssignVariableOpconv2d_1/kerneltraining/Adam/sub_10*
dtype0
?
training/Adam/ReadVariableOp_25ReadVariableOpconv2d_1/kernel!^training/Adam/AssignVariableOp_8*
dtype0*&
_output_shapes
:  
c
training/Adam/ReadVariableOp_26ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
x
#training/Adam/mul_16/ReadVariableOpReadVariableOptraining/Adam/Variable_3*
dtype0*
_output_shapes
: 
?
training/Adam/mul_16Multraining/Adam/ReadVariableOp_26#training/Adam/mul_16/ReadVariableOp*
T0*
_output_shapes
: 
c
training/Adam/ReadVariableOp_27ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_11/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
u
training/Adam/sub_11Subtraining/Adam/sub_11/xtraining/Adam/ReadVariableOp_27*
T0*
_output_shapes
: 
?
training/Adam/mul_17Multraining/Adam/sub_119training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
l
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
T0*
_output_shapes
: 
c
training/Adam/ReadVariableOp_28ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_18/ReadVariableOpReadVariableOptraining/Adam/Variable_15*
dtype0*
_output_shapes
: 
?
training/Adam/mul_18Multraining/Adam/ReadVariableOp_28#training/Adam/mul_18/ReadVariableOp*
_output_shapes
: *
T0
c
training/Adam/ReadVariableOp_29ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_12/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
u
training/Adam/sub_12Subtraining/Adam/sub_12/xtraining/Adam/ReadVariableOp_29*
_output_shapes
: *
T0
?
training/Adam/Square_3Square9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
n
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
T0*
_output_shapes
: 
l
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes
: 
i
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0*
_output_shapes
: 
Z
training/Adam/Const_9Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_10Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
?
%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_10*
T0*
_output_shapes
: 
?
training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_9*
T0*
_output_shapes
: 
`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
_output_shapes
: *
T0
[
training/Adam/add_12/yConst*
valueB
 *???3*
dtype0*
_output_shapes
: 
n
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
T0*
_output_shapes
: 
s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
_output_shapes
: *
T0
i
training/Adam/ReadVariableOp_30ReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
: 
z
training/Adam/sub_13Subtraining/Adam/ReadVariableOp_30training/Adam/truediv_4*
T0*
_output_shapes
: 
q
 training/Adam/AssignVariableOp_9AssignVariableOptraining/Adam/Variable_3training/Adam/add_10*
dtype0
?
training/Adam/ReadVariableOp_31ReadVariableOptraining/Adam/Variable_3!^training/Adam/AssignVariableOp_9*
dtype0*
_output_shapes
: 
s
!training/Adam/AssignVariableOp_10AssignVariableOptraining/Adam/Variable_15training/Adam/add_11*
dtype0
?
training/Adam/ReadVariableOp_32ReadVariableOptraining/Adam/Variable_15"^training/Adam/AssignVariableOp_10*
dtype0*
_output_shapes
: 
g
!training/Adam/AssignVariableOp_11AssignVariableOpconv2d_1/biastraining/Adam/sub_13*
dtype0
?
training/Adam/ReadVariableOp_33ReadVariableOpconv2d_1/bias"^training/Adam/AssignVariableOp_11*
dtype0*
_output_shapes
: 
c
training/Adam/ReadVariableOp_34ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 

#training/Adam/mul_21/ReadVariableOpReadVariableOptraining/Adam/Variable_4*
dtype0*!
_output_shapes
:???
?
training/Adam/mul_21Multraining/Adam/ReadVariableOp_34#training/Adam/mul_21/ReadVariableOp*!
_output_shapes
:???*
T0
c
training/Adam/ReadVariableOp_35ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_14/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
u
training/Adam/sub_14Subtraining/Adam/sub_14/xtraining/Adam/ReadVariableOp_35*
T0*
_output_shapes
: 
?
training/Adam/mul_22Multraining/Adam/sub_142training/Adam/gradients/dense/MatMul_grad/MatMul_1*
T0*!
_output_shapes
:???
s
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
T0*!
_output_shapes
:???
c
training/Adam/ReadVariableOp_36ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
?
#training/Adam/mul_23/ReadVariableOpReadVariableOptraining/Adam/Variable_16*
dtype0*!
_output_shapes
:???
?
training/Adam/mul_23Multraining/Adam/ReadVariableOp_36#training/Adam/mul_23/ReadVariableOp*
T0*!
_output_shapes
:???
c
training/Adam/ReadVariableOp_37ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_15/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
u
training/Adam/sub_15Subtraining/Adam/sub_15/xtraining/Adam/ReadVariableOp_37*
_output_shapes
: *
T0
?
training/Adam/Square_4Square2training/Adam/gradients/dense/MatMul_grad/MatMul_1*
T0*!
_output_shapes
:???
u
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0*!
_output_shapes
:???
s
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
T0*!
_output_shapes
:???
p
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
T0*!
_output_shapes
:???
[
training/Adam/Const_11Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_12Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
?
%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_12*
T0*!
_output_shapes
:???
?
training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_11*
T0*!
_output_shapes
:???
g
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0*!
_output_shapes
:???
[
training/Adam/add_15/yConst*
valueB
 *???3*
dtype0*
_output_shapes
: 
u
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
T0*!
_output_shapes
:???
z
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
T0*!
_output_shapes
:???
o
training/Adam/ReadVariableOp_38ReadVariableOpdense/kernel*
dtype0*!
_output_shapes
:???
?
training/Adam/sub_16Subtraining/Adam/ReadVariableOp_38training/Adam/truediv_5*!
_output_shapes
:???*
T0
r
!training/Adam/AssignVariableOp_12AssignVariableOptraining/Adam/Variable_4training/Adam/add_13*
dtype0
?
training/Adam/ReadVariableOp_39ReadVariableOptraining/Adam/Variable_4"^training/Adam/AssignVariableOp_12*
dtype0*!
_output_shapes
:???
s
!training/Adam/AssignVariableOp_13AssignVariableOptraining/Adam/Variable_16training/Adam/add_14*
dtype0
?
training/Adam/ReadVariableOp_40ReadVariableOptraining/Adam/Variable_16"^training/Adam/AssignVariableOp_13*
dtype0*!
_output_shapes
:???
f
!training/Adam/AssignVariableOp_14AssignVariableOpdense/kerneltraining/Adam/sub_16*
dtype0
?
training/Adam/ReadVariableOp_41ReadVariableOpdense/kernel"^training/Adam/AssignVariableOp_14*
dtype0*!
_output_shapes
:???
c
training/Adam/ReadVariableOp_42ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_26/ReadVariableOpReadVariableOptraining/Adam/Variable_5*
dtype0*
_output_shapes	
:?
?
training/Adam/mul_26Multraining/Adam/ReadVariableOp_42#training/Adam/mul_26/ReadVariableOp*
T0*
_output_shapes	
:?
c
training/Adam/ReadVariableOp_43ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_17/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
u
training/Adam/sub_17Subtraining/Adam/sub_17/xtraining/Adam/ReadVariableOp_43*
_output_shapes
: *
T0
?
training/Adam/mul_27Multraining/Adam/sub_176training/Adam/gradients/dense/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:?
m
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
_output_shapes	
:?*
T0
c
training/Adam/ReadVariableOp_44ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
z
#training/Adam/mul_28/ReadVariableOpReadVariableOptraining/Adam/Variable_17*
dtype0*
_output_shapes	
:?
?
training/Adam/mul_28Multraining/Adam/ReadVariableOp_44#training/Adam/mul_28/ReadVariableOp*
T0*
_output_shapes	
:?
c
training/Adam/ReadVariableOp_45ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_18/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
u
training/Adam/sub_18Subtraining/Adam/sub_18/xtraining/Adam/ReadVariableOp_45*
T0*
_output_shapes
: 
~
training/Adam/Square_5Square6training/Adam/gradients/dense/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:?
o
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
T0*
_output_shapes	
:?
m
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
T0*
_output_shapes	
:?
j
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
_output_shapes	
:?*
T0
[
training/Adam/Const_13Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_14Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
?
%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_14*
T0*
_output_shapes	
:?
?
training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_13*
T0*
_output_shapes	
:?
a
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
_output_shapes	
:?*
T0
[
training/Adam/add_18/yConst*
valueB
 *???3*
dtype0*
_output_shapes
: 
o
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
T0*
_output_shapes	
:?
t
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
T0*
_output_shapes	
:?
g
training/Adam/ReadVariableOp_46ReadVariableOp
dense/bias*
dtype0*
_output_shapes	
:?
{
training/Adam/sub_19Subtraining/Adam/ReadVariableOp_46training/Adam/truediv_6*
T0*
_output_shapes	
:?
r
!training/Adam/AssignVariableOp_15AssignVariableOptraining/Adam/Variable_5training/Adam/add_16*
dtype0
?
training/Adam/ReadVariableOp_47ReadVariableOptraining/Adam/Variable_5"^training/Adam/AssignVariableOp_15*
dtype0*
_output_shapes	
:?
s
!training/Adam/AssignVariableOp_16AssignVariableOptraining/Adam/Variable_17training/Adam/add_17*
dtype0
?
training/Adam/ReadVariableOp_48ReadVariableOptraining/Adam/Variable_17"^training/Adam/AssignVariableOp_16*
dtype0*
_output_shapes	
:?
d
!training/Adam/AssignVariableOp_17AssignVariableOp
dense/biastraining/Adam/sub_19*
dtype0
?
training/Adam/ReadVariableOp_49ReadVariableOp
dense/bias"^training/Adam/AssignVariableOp_17*
_output_shapes	
:?*
dtype0
c
training/Adam/ReadVariableOp_50ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
~
#training/Adam/mul_31/ReadVariableOpReadVariableOptraining/Adam/Variable_6*
dtype0* 
_output_shapes
:
??
?
training/Adam/mul_31Multraining/Adam/ReadVariableOp_50#training/Adam/mul_31/ReadVariableOp*
T0* 
_output_shapes
:
??
c
training/Adam/ReadVariableOp_51ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_20/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
u
training/Adam/sub_20Subtraining/Adam/sub_20/xtraining/Adam/ReadVariableOp_51*
T0*
_output_shapes
: 
?
training/Adam/mul_32Multraining/Adam/sub_204training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
??
r
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32* 
_output_shapes
:
??*
T0
c
training/Adam/ReadVariableOp_52ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 

#training/Adam/mul_33/ReadVariableOpReadVariableOptraining/Adam/Variable_18*
dtype0* 
_output_shapes
:
??
?
training/Adam/mul_33Multraining/Adam/ReadVariableOp_52#training/Adam/mul_33/ReadVariableOp*
T0* 
_output_shapes
:
??
c
training/Adam/ReadVariableOp_53ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_21/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
u
training/Adam/sub_21Subtraining/Adam/sub_21/xtraining/Adam/ReadVariableOp_53*
T0*
_output_shapes
: 
?
training/Adam/Square_6Square4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:
??*
T0
t
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6* 
_output_shapes
:
??*
T0
r
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
T0* 
_output_shapes
:
??
o
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0* 
_output_shapes
:
??
[
training/Adam/Const_15Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_16Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
?
%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_16*
T0* 
_output_shapes
:
??
?
training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_15*
T0* 
_output_shapes
:
??
f
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
T0* 
_output_shapes
:
??
[
training/Adam/add_21/yConst*
valueB
 *???3*
dtype0*
_output_shapes
: 
t
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y* 
_output_shapes
:
??*
T0
y
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0* 
_output_shapes
:
??
p
training/Adam/ReadVariableOp_54ReadVariableOpdense_1/kernel*
dtype0* 
_output_shapes
:
??
?
training/Adam/sub_22Subtraining/Adam/ReadVariableOp_54training/Adam/truediv_7*
T0* 
_output_shapes
:
??
r
!training/Adam/AssignVariableOp_18AssignVariableOptraining/Adam/Variable_6training/Adam/add_19*
dtype0
?
training/Adam/ReadVariableOp_55ReadVariableOptraining/Adam/Variable_6"^training/Adam/AssignVariableOp_18*
dtype0* 
_output_shapes
:
??
s
!training/Adam/AssignVariableOp_19AssignVariableOptraining/Adam/Variable_18training/Adam/add_20*
dtype0
?
training/Adam/ReadVariableOp_56ReadVariableOptraining/Adam/Variable_18"^training/Adam/AssignVariableOp_19*
dtype0* 
_output_shapes
:
??
h
!training/Adam/AssignVariableOp_20AssignVariableOpdense_1/kerneltraining/Adam/sub_22*
dtype0
?
training/Adam/ReadVariableOp_57ReadVariableOpdense_1/kernel"^training/Adam/AssignVariableOp_20*
dtype0* 
_output_shapes
:
??
c
training/Adam/ReadVariableOp_58ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_36/ReadVariableOpReadVariableOptraining/Adam/Variable_7*
dtype0*
_output_shapes	
:?
?
training/Adam/mul_36Multraining/Adam/ReadVariableOp_58#training/Adam/mul_36/ReadVariableOp*
_output_shapes	
:?*
T0
c
training/Adam/ReadVariableOp_59ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_23/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
u
training/Adam/sub_23Subtraining/Adam/sub_23/xtraining/Adam/ReadVariableOp_59*
_output_shapes
: *
T0
?
training/Adam/mul_37Multraining/Adam/sub_238training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?*
T0
m
training/Adam/add_22Addtraining/Adam/mul_36training/Adam/mul_37*
_output_shapes	
:?*
T0
c
training/Adam/ReadVariableOp_60ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
z
#training/Adam/mul_38/ReadVariableOpReadVariableOptraining/Adam/Variable_19*
dtype0*
_output_shapes	
:?
?
training/Adam/mul_38Multraining/Adam/ReadVariableOp_60#training/Adam/mul_38/ReadVariableOp*
T0*
_output_shapes	
:?
c
training/Adam/ReadVariableOp_61ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_24/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
u
training/Adam/sub_24Subtraining/Adam/sub_24/xtraining/Adam/ReadVariableOp_61*
_output_shapes
: *
T0
?
training/Adam/Square_7Square8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:?
o
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
T0*
_output_shapes	
:?
m
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
T0*
_output_shapes	
:?
j
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
T0*
_output_shapes	
:?
[
training/Adam/Const_17Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_18Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
?
%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_18*
T0*
_output_shapes	
:?
?
training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_17*
T0*
_output_shapes	
:?
a
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
T0*
_output_shapes	
:?
[
training/Adam/add_24/yConst*
valueB
 *???3*
dtype0*
_output_shapes
: 
o
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y*
_output_shapes	
:?*
T0
t
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
T0*
_output_shapes	
:?
i
training/Adam/ReadVariableOp_62ReadVariableOpdense_1/bias*
dtype0*
_output_shapes	
:?
{
training/Adam/sub_25Subtraining/Adam/ReadVariableOp_62training/Adam/truediv_8*
T0*
_output_shapes	
:?
r
!training/Adam/AssignVariableOp_21AssignVariableOptraining/Adam/Variable_7training/Adam/add_22*
dtype0
?
training/Adam/ReadVariableOp_63ReadVariableOptraining/Adam/Variable_7"^training/Adam/AssignVariableOp_21*
dtype0*
_output_shapes	
:?
s
!training/Adam/AssignVariableOp_22AssignVariableOptraining/Adam/Variable_19training/Adam/add_23*
dtype0
?
training/Adam/ReadVariableOp_64ReadVariableOptraining/Adam/Variable_19"^training/Adam/AssignVariableOp_22*
dtype0*
_output_shapes	
:?
f
!training/Adam/AssignVariableOp_23AssignVariableOpdense_1/biastraining/Adam/sub_25*
dtype0
?
training/Adam/ReadVariableOp_65ReadVariableOpdense_1/bias"^training/Adam/AssignVariableOp_23*
dtype0*
_output_shapes	
:?
c
training/Adam/ReadVariableOp_66ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
}
#training/Adam/mul_41/ReadVariableOpReadVariableOptraining/Adam/Variable_8*
dtype0*
_output_shapes
:	?
?
training/Adam/mul_41Multraining/Adam/ReadVariableOp_66#training/Adam/mul_41/ReadVariableOp*
T0*
_output_shapes
:	?
c
training/Adam/ReadVariableOp_67ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_26/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
u
training/Adam/sub_26Subtraining/Adam/sub_26/xtraining/Adam/ReadVariableOp_67*
T0*
_output_shapes
: 
?
training/Adam/mul_42Multraining/Adam/sub_264training/Adam/gradients/weather/MatMul_grad/MatMul_1*
_output_shapes
:	?*
T0
q
training/Adam/add_25Addtraining/Adam/mul_41training/Adam/mul_42*
_output_shapes
:	?*
T0
c
training/Adam/ReadVariableOp_68ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
~
#training/Adam/mul_43/ReadVariableOpReadVariableOptraining/Adam/Variable_20*
dtype0*
_output_shapes
:	?
?
training/Adam/mul_43Multraining/Adam/ReadVariableOp_68#training/Adam/mul_43/ReadVariableOp*
T0*
_output_shapes
:	?
c
training/Adam/ReadVariableOp_69ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_27/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
u
training/Adam/sub_27Subtraining/Adam/sub_27/xtraining/Adam/ReadVariableOp_69*
T0*
_output_shapes
: 
?
training/Adam/Square_8Square4training/Adam/gradients/weather/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	?
s
training/Adam/mul_44Multraining/Adam/sub_27training/Adam/Square_8*
T0*
_output_shapes
:	?
q
training/Adam/add_26Addtraining/Adam/mul_43training/Adam/mul_44*
T0*
_output_shapes
:	?
n
training/Adam/mul_45Multraining/Adam/multraining/Adam/add_25*
_output_shapes
:	?*
T0
[
training/Adam/Const_19Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_20Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
?
%training/Adam/clip_by_value_9/MinimumMinimumtraining/Adam/add_26training/Adam/Const_20*
T0*
_output_shapes
:	?
?
training/Adam/clip_by_value_9Maximum%training/Adam/clip_by_value_9/Minimumtraining/Adam/Const_19*
_output_shapes
:	?*
T0
e
training/Adam/Sqrt_9Sqrttraining/Adam/clip_by_value_9*
T0*
_output_shapes
:	?
[
training/Adam/add_27/yConst*
valueB
 *???3*
dtype0*
_output_shapes
: 
s
training/Adam/add_27Addtraining/Adam/Sqrt_9training/Adam/add_27/y*
T0*
_output_shapes
:	?
x
training/Adam/truediv_9RealDivtraining/Adam/mul_45training/Adam/add_27*
T0*
_output_shapes
:	?
o
training/Adam/ReadVariableOp_70ReadVariableOpweather/kernel*
dtype0*
_output_shapes
:	?

training/Adam/sub_28Subtraining/Adam/ReadVariableOp_70training/Adam/truediv_9*
T0*
_output_shapes
:	?
r
!training/Adam/AssignVariableOp_24AssignVariableOptraining/Adam/Variable_8training/Adam/add_25*
dtype0
?
training/Adam/ReadVariableOp_71ReadVariableOptraining/Adam/Variable_8"^training/Adam/AssignVariableOp_24*
dtype0*
_output_shapes
:	?
s
!training/Adam/AssignVariableOp_25AssignVariableOptraining/Adam/Variable_20training/Adam/add_26*
dtype0
?
training/Adam/ReadVariableOp_72ReadVariableOptraining/Adam/Variable_20"^training/Adam/AssignVariableOp_25*
dtype0*
_output_shapes
:	?
h
!training/Adam/AssignVariableOp_26AssignVariableOpweather/kerneltraining/Adam/sub_28*
dtype0
?
training/Adam/ReadVariableOp_73ReadVariableOpweather/kernel"^training/Adam/AssignVariableOp_26*
dtype0*
_output_shapes
:	?
c
training/Adam/ReadVariableOp_74ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
x
#training/Adam/mul_46/ReadVariableOpReadVariableOptraining/Adam/Variable_9*
dtype0*
_output_shapes
:
?
training/Adam/mul_46Multraining/Adam/ReadVariableOp_74#training/Adam/mul_46/ReadVariableOp*
_output_shapes
:*
T0
c
training/Adam/ReadVariableOp_75ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_29/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
u
training/Adam/sub_29Subtraining/Adam/sub_29/xtraining/Adam/ReadVariableOp_75*
T0*
_output_shapes
: 
?
training/Adam/mul_47Multraining/Adam/sub_298training/Adam/gradients/weather/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training/Adam/add_28Addtraining/Adam/mul_46training/Adam/mul_47*
T0*
_output_shapes
:
c
training/Adam/ReadVariableOp_76ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_48/ReadVariableOpReadVariableOptraining/Adam/Variable_21*
dtype0*
_output_shapes
:
?
training/Adam/mul_48Multraining/Adam/ReadVariableOp_76#training/Adam/mul_48/ReadVariableOp*
T0*
_output_shapes
:
c
training/Adam/ReadVariableOp_77ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_30/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
u
training/Adam/sub_30Subtraining/Adam/sub_30/xtraining/Adam/ReadVariableOp_77*
T0*
_output_shapes
: 

training/Adam/Square_9Square8training/Adam/gradients/weather/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
n
training/Adam/mul_49Multraining/Adam/sub_30training/Adam/Square_9*
_output_shapes
:*
T0
l
training/Adam/add_29Addtraining/Adam/mul_48training/Adam/mul_49*
T0*
_output_shapes
:
i
training/Adam/mul_50Multraining/Adam/multraining/Adam/add_28*
T0*
_output_shapes
:
[
training/Adam/Const_21Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_22Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
?
&training/Adam/clip_by_value_10/MinimumMinimumtraining/Adam/add_29training/Adam/Const_22*
_output_shapes
:*
T0
?
training/Adam/clip_by_value_10Maximum&training/Adam/clip_by_value_10/Minimumtraining/Adam/Const_21*
T0*
_output_shapes
:
b
training/Adam/Sqrt_10Sqrttraining/Adam/clip_by_value_10*
_output_shapes
:*
T0
[
training/Adam/add_30/yConst*
valueB
 *???3*
dtype0*
_output_shapes
: 
o
training/Adam/add_30Addtraining/Adam/Sqrt_10training/Adam/add_30/y*
_output_shapes
:*
T0
t
training/Adam/truediv_10RealDivtraining/Adam/mul_50training/Adam/add_30*
T0*
_output_shapes
:
h
training/Adam/ReadVariableOp_78ReadVariableOpweather/bias*
dtype0*
_output_shapes
:
{
training/Adam/sub_31Subtraining/Adam/ReadVariableOp_78training/Adam/truediv_10*
_output_shapes
:*
T0
r
!training/Adam/AssignVariableOp_27AssignVariableOptraining/Adam/Variable_9training/Adam/add_28*
dtype0
?
training/Adam/ReadVariableOp_79ReadVariableOptraining/Adam/Variable_9"^training/Adam/AssignVariableOp_27*
dtype0*
_output_shapes
:
s
!training/Adam/AssignVariableOp_28AssignVariableOptraining/Adam/Variable_21training/Adam/add_29*
dtype0
?
training/Adam/ReadVariableOp_80ReadVariableOptraining/Adam/Variable_21"^training/Adam/AssignVariableOp_28*
dtype0*
_output_shapes
:
f
!training/Adam/AssignVariableOp_29AssignVariableOpweather/biastraining/Adam/sub_31*
dtype0
?
training/Adam/ReadVariableOp_81ReadVariableOpweather/bias"^training/Adam/AssignVariableOp_29*
dtype0*
_output_shapes
:
c
training/Adam/ReadVariableOp_82ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
~
#training/Adam/mul_51/ReadVariableOpReadVariableOptraining/Adam/Variable_10*
dtype0*
_output_shapes
:	?
?
training/Adam/mul_51Multraining/Adam/ReadVariableOp_82#training/Adam/mul_51/ReadVariableOp*
T0*
_output_shapes
:	?
c
training/Adam/ReadVariableOp_83ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_32/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
u
training/Adam/sub_32Subtraining/Adam/sub_32/xtraining/Adam/ReadVariableOp_83*
T0*
_output_shapes
: 
?
training/Adam/mul_52Multraining/Adam/sub_323training/Adam/gradients/ground/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	?
q
training/Adam/add_31Addtraining/Adam/mul_51training/Adam/mul_52*
T0*
_output_shapes
:	?
c
training/Adam/ReadVariableOp_84ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
~
#training/Adam/mul_53/ReadVariableOpReadVariableOptraining/Adam/Variable_22*
dtype0*
_output_shapes
:	?
?
training/Adam/mul_53Multraining/Adam/ReadVariableOp_84#training/Adam/mul_53/ReadVariableOp*
_output_shapes
:	?*
T0
c
training/Adam/ReadVariableOp_85ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_33/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
u
training/Adam/sub_33Subtraining/Adam/sub_33/xtraining/Adam/ReadVariableOp_85*
_output_shapes
: *
T0
?
training/Adam/Square_10Square3training/Adam/gradients/ground/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	?
t
training/Adam/mul_54Multraining/Adam/sub_33training/Adam/Square_10*
_output_shapes
:	?*
T0
q
training/Adam/add_32Addtraining/Adam/mul_53training/Adam/mul_54*
T0*
_output_shapes
:	?
n
training/Adam/mul_55Multraining/Adam/multraining/Adam/add_31*
_output_shapes
:	?*
T0
[
training/Adam/Const_23Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_24Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
?
&training/Adam/clip_by_value_11/MinimumMinimumtraining/Adam/add_32training/Adam/Const_24*
_output_shapes
:	?*
T0
?
training/Adam/clip_by_value_11Maximum&training/Adam/clip_by_value_11/Minimumtraining/Adam/Const_23*
T0*
_output_shapes
:	?
g
training/Adam/Sqrt_11Sqrttraining/Adam/clip_by_value_11*
T0*
_output_shapes
:	?
[
training/Adam/add_33/yConst*
valueB
 *???3*
dtype0*
_output_shapes
: 
t
training/Adam/add_33Addtraining/Adam/Sqrt_11training/Adam/add_33/y*
T0*
_output_shapes
:	?
y
training/Adam/truediv_11RealDivtraining/Adam/mul_55training/Adam/add_33*
_output_shapes
:	?*
T0
n
training/Adam/ReadVariableOp_86ReadVariableOpground/kernel*
dtype0*
_output_shapes
:	?
?
training/Adam/sub_34Subtraining/Adam/ReadVariableOp_86training/Adam/truediv_11*
_output_shapes
:	?*
T0
s
!training/Adam/AssignVariableOp_30AssignVariableOptraining/Adam/Variable_10training/Adam/add_31*
dtype0
?
training/Adam/ReadVariableOp_87ReadVariableOptraining/Adam/Variable_10"^training/Adam/AssignVariableOp_30*
dtype0*
_output_shapes
:	?
s
!training/Adam/AssignVariableOp_31AssignVariableOptraining/Adam/Variable_22training/Adam/add_32*
dtype0
?
training/Adam/ReadVariableOp_88ReadVariableOptraining/Adam/Variable_22"^training/Adam/AssignVariableOp_31*
dtype0*
_output_shapes
:	?
g
!training/Adam/AssignVariableOp_32AssignVariableOpground/kerneltraining/Adam/sub_34*
dtype0
?
training/Adam/ReadVariableOp_89ReadVariableOpground/kernel"^training/Adam/AssignVariableOp_32*
dtype0*
_output_shapes
:	?
c
training/Adam/ReadVariableOp_90ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_56/ReadVariableOpReadVariableOptraining/Adam/Variable_11*
dtype0*
_output_shapes
:
?
training/Adam/mul_56Multraining/Adam/ReadVariableOp_90#training/Adam/mul_56/ReadVariableOp*
_output_shapes
:*
T0
c
training/Adam/ReadVariableOp_91ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_35/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
u
training/Adam/sub_35Subtraining/Adam/sub_35/xtraining/Adam/ReadVariableOp_91*
_output_shapes
: *
T0
?
training/Adam/mul_57Multraining/Adam/sub_357training/Adam/gradients/ground/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
l
training/Adam/add_34Addtraining/Adam/mul_56training/Adam/mul_57*
T0*
_output_shapes
:
c
training/Adam/ReadVariableOp_92ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
y
#training/Adam/mul_58/ReadVariableOpReadVariableOptraining/Adam/Variable_23*
dtype0*
_output_shapes
:
?
training/Adam/mul_58Multraining/Adam/ReadVariableOp_92#training/Adam/mul_58/ReadVariableOp*
T0*
_output_shapes
:
c
training/Adam/ReadVariableOp_93ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_36/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
u
training/Adam/sub_36Subtraining/Adam/sub_36/xtraining/Adam/ReadVariableOp_93*
T0*
_output_shapes
: 

training/Adam/Square_11Square7training/Adam/gradients/ground/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
o
training/Adam/mul_59Multraining/Adam/sub_36training/Adam/Square_11*
T0*
_output_shapes
:
l
training/Adam/add_35Addtraining/Adam/mul_58training/Adam/mul_59*
T0*
_output_shapes
:
i
training/Adam/mul_60Multraining/Adam/multraining/Adam/add_34*
T0*
_output_shapes
:
[
training/Adam/Const_25Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_26Const*
_output_shapes
: *
valueB
 *  ?*
dtype0
?
&training/Adam/clip_by_value_12/MinimumMinimumtraining/Adam/add_35training/Adam/Const_26*
_output_shapes
:*
T0
?
training/Adam/clip_by_value_12Maximum&training/Adam/clip_by_value_12/Minimumtraining/Adam/Const_25*
T0*
_output_shapes
:
b
training/Adam/Sqrt_12Sqrttraining/Adam/clip_by_value_12*
T0*
_output_shapes
:
[
training/Adam/add_36/yConst*
valueB
 *???3*
dtype0*
_output_shapes
: 
o
training/Adam/add_36Addtraining/Adam/Sqrt_12training/Adam/add_36/y*
T0*
_output_shapes
:
t
training/Adam/truediv_12RealDivtraining/Adam/mul_60training/Adam/add_36*
T0*
_output_shapes
:
g
training/Adam/ReadVariableOp_94ReadVariableOpground/bias*
dtype0*
_output_shapes
:
{
training/Adam/sub_37Subtraining/Adam/ReadVariableOp_94training/Adam/truediv_12*
_output_shapes
:*
T0
s
!training/Adam/AssignVariableOp_33AssignVariableOptraining/Adam/Variable_11training/Adam/add_34*
dtype0
?
training/Adam/ReadVariableOp_95ReadVariableOptraining/Adam/Variable_11"^training/Adam/AssignVariableOp_33*
dtype0*
_output_shapes
:
s
!training/Adam/AssignVariableOp_34AssignVariableOptraining/Adam/Variable_23training/Adam/add_35*
dtype0
?
training/Adam/ReadVariableOp_96ReadVariableOptraining/Adam/Variable_23"^training/Adam/AssignVariableOp_34*
dtype0*
_output_shapes
:
e
!training/Adam/AssignVariableOp_35AssignVariableOpground/biastraining/Adam/sub_37*
dtype0
?
training/Adam/ReadVariableOp_97ReadVariableOpground/bias"^training/Adam/AssignVariableOp_35*
dtype0*
_output_shapes
:
?

training/group_depsNoOp	^loss/add^loss/ground_loss/Mean_3^loss/weather_loss/Mean_2^training/Adam/ReadVariableOp ^training/Adam/ReadVariableOp_15 ^training/Adam/ReadVariableOp_16 ^training/Adam/ReadVariableOp_17 ^training/Adam/ReadVariableOp_23 ^training/Adam/ReadVariableOp_24 ^training/Adam/ReadVariableOp_25 ^training/Adam/ReadVariableOp_31 ^training/Adam/ReadVariableOp_32 ^training/Adam/ReadVariableOp_33 ^training/Adam/ReadVariableOp_39 ^training/Adam/ReadVariableOp_40 ^training/Adam/ReadVariableOp_41 ^training/Adam/ReadVariableOp_47 ^training/Adam/ReadVariableOp_48 ^training/Adam/ReadVariableOp_49 ^training/Adam/ReadVariableOp_55 ^training/Adam/ReadVariableOp_56 ^training/Adam/ReadVariableOp_57 ^training/Adam/ReadVariableOp_63 ^training/Adam/ReadVariableOp_64 ^training/Adam/ReadVariableOp_65^training/Adam/ReadVariableOp_7 ^training/Adam/ReadVariableOp_71 ^training/Adam/ReadVariableOp_72 ^training/Adam/ReadVariableOp_73 ^training/Adam/ReadVariableOp_79^training/Adam/ReadVariableOp_8 ^training/Adam/ReadVariableOp_80 ^training/Adam/ReadVariableOp_81 ^training/Adam/ReadVariableOp_87 ^training/Adam/ReadVariableOp_88 ^training/Adam/ReadVariableOp_89^training/Adam/ReadVariableOp_9 ^training/Adam/ReadVariableOp_95 ^training/Adam/ReadVariableOp_96 ^training/Adam/ReadVariableOp_97
T
VarIsInitializedOp_12VarIsInitializedOpAdam/iterations*
_output_shapes
: 
L
VarIsInitializedOp_13VarIsInitializedOpAdam/lr*
_output_shapes
: 
P
VarIsInitializedOp_14VarIsInitializedOpAdam/beta_1*
_output_shapes
: 
P
VarIsInitializedOp_15VarIsInitializedOpAdam/beta_2*
_output_shapes
: 
O
VarIsInitializedOp_16VarIsInitializedOp
Adam/decay*
_output_shapes
: 
[
VarIsInitializedOp_17VarIsInitializedOptraining/Adam/Variable*
_output_shapes
: 
]
VarIsInitializedOp_18VarIsInitializedOptraining/Adam/Variable_1*
_output_shapes
: 
]
VarIsInitializedOp_19VarIsInitializedOptraining/Adam/Variable_2*
_output_shapes
: 
]
VarIsInitializedOp_20VarIsInitializedOptraining/Adam/Variable_3*
_output_shapes
: 
]
VarIsInitializedOp_21VarIsInitializedOptraining/Adam/Variable_4*
_output_shapes
: 
]
VarIsInitializedOp_22VarIsInitializedOptraining/Adam/Variable_5*
_output_shapes
: 
]
VarIsInitializedOp_23VarIsInitializedOptraining/Adam/Variable_6*
_output_shapes
: 
]
VarIsInitializedOp_24VarIsInitializedOptraining/Adam/Variable_7*
_output_shapes
: 
]
VarIsInitializedOp_25VarIsInitializedOptraining/Adam/Variable_8*
_output_shapes
: 
]
VarIsInitializedOp_26VarIsInitializedOptraining/Adam/Variable_9*
_output_shapes
: 
^
VarIsInitializedOp_27VarIsInitializedOptraining/Adam/Variable_10*
_output_shapes
: 
^
VarIsInitializedOp_28VarIsInitializedOptraining/Adam/Variable_11*
_output_shapes
: 
^
VarIsInitializedOp_29VarIsInitializedOptraining/Adam/Variable_12*
_output_shapes
: 
^
VarIsInitializedOp_30VarIsInitializedOptraining/Adam/Variable_13*
_output_shapes
: 
^
VarIsInitializedOp_31VarIsInitializedOptraining/Adam/Variable_14*
_output_shapes
: 
^
VarIsInitializedOp_32VarIsInitializedOptraining/Adam/Variable_15*
_output_shapes
: 
^
VarIsInitializedOp_33VarIsInitializedOptraining/Adam/Variable_16*
_output_shapes
: 
^
VarIsInitializedOp_34VarIsInitializedOptraining/Adam/Variable_17*
_output_shapes
: 
^
VarIsInitializedOp_35VarIsInitializedOptraining/Adam/Variable_18*
_output_shapes
: 
^
VarIsInitializedOp_36VarIsInitializedOptraining/Adam/Variable_19*
_output_shapes
: 
^
VarIsInitializedOp_37VarIsInitializedOptraining/Adam/Variable_20*
_output_shapes
: 
^
VarIsInitializedOp_38VarIsInitializedOptraining/Adam/Variable_21*
_output_shapes
: 
^
VarIsInitializedOp_39VarIsInitializedOptraining/Adam/Variable_22*
_output_shapes
: 
^
VarIsInitializedOp_40VarIsInitializedOptraining/Adam/Variable_23*
_output_shapes
: 
^
VarIsInitializedOp_41VarIsInitializedOptraining/Adam/Variable_24*
_output_shapes
: 
^
VarIsInitializedOp_42VarIsInitializedOptraining/Adam/Variable_25*
_output_shapes
: 
^
VarIsInitializedOp_43VarIsInitializedOptraining/Adam/Variable_26*
_output_shapes
: 
^
VarIsInitializedOp_44VarIsInitializedOptraining/Adam/Variable_27*
_output_shapes
: 
^
VarIsInitializedOp_45VarIsInitializedOptraining/Adam/Variable_28*
_output_shapes
: 
^
VarIsInitializedOp_46VarIsInitializedOptraining/Adam/Variable_29*
_output_shapes
: 
^
VarIsInitializedOp_47VarIsInitializedOptraining/Adam/Variable_30*
_output_shapes
: 
^
VarIsInitializedOp_48VarIsInitializedOptraining/Adam/Variable_31*
_output_shapes
: 
^
VarIsInitializedOp_49VarIsInitializedOptraining/Adam/Variable_32*
_output_shapes
: 
^
VarIsInitializedOp_50VarIsInitializedOptraining/Adam/Variable_33*
_output_shapes
: 
^
VarIsInitializedOp_51VarIsInitializedOptraining/Adam/Variable_34*
_output_shapes
: 
^
VarIsInitializedOp_52VarIsInitializedOptraining/Adam/Variable_35*
_output_shapes
: 
?

init_1NoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign!^training/Adam/Variable_24/Assign!^training/Adam/Variable_25/Assign!^training/Adam/Variable_26/Assign!^training/Adam/Variable_27/Assign!^training/Adam/Variable_28/Assign!^training/Adam/Variable_29/Assign ^training/Adam/Variable_3/Assign!^training/Adam/Variable_30/Assign!^training/Adam/Variable_31/Assign!^training/Adam/Variable_32/Assign!^training/Adam/Variable_33/Assign!^training/Adam/Variable_34/Assign!^training/Adam/Variable_35/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign
O
Placeholder_12Placeholder*
dtype0	*
_output_shapes
: *
shape: 
U
AssignVariableOp_12AssignVariableOpAdam/iterationsPlaceholder_12*
dtype0	
o
ReadVariableOp_12ReadVariableOpAdam/iterations^AssignVariableOp_12*
dtype0	*
_output_shapes
: 
o
Placeholder_13Placeholder*
shape: *
dtype0*&
_output_shapes
: 
\
AssignVariableOp_13AssignVariableOptraining/Adam/VariablePlaceholder_13*
dtype0
?
ReadVariableOp_13ReadVariableOptraining/Adam/Variable^AssignVariableOp_13*
dtype0*&
_output_shapes
: 
W
Placeholder_14Placeholder*
dtype0*
_output_shapes
: *
shape: 
^
AssignVariableOp_14AssignVariableOptraining/Adam/Variable_1Placeholder_14*
dtype0
|
ReadVariableOp_14ReadVariableOptraining/Adam/Variable_1^AssignVariableOp_14*
dtype0*
_output_shapes
: 
o
Placeholder_15Placeholder*
dtype0*&
_output_shapes
:  *
shape:  
^
AssignVariableOp_15AssignVariableOptraining/Adam/Variable_2Placeholder_15*
dtype0
?
ReadVariableOp_15ReadVariableOptraining/Adam/Variable_2^AssignVariableOp_15*
dtype0*&
_output_shapes
:  
W
Placeholder_16Placeholder*
dtype0*
_output_shapes
: *
shape: 
^
AssignVariableOp_16AssignVariableOptraining/Adam/Variable_3Placeholder_16*
dtype0
|
ReadVariableOp_16ReadVariableOptraining/Adam/Variable_3^AssignVariableOp_16*
dtype0*
_output_shapes
: 
e
Placeholder_17Placeholder*
dtype0*!
_output_shapes
:???*
shape:???
^
AssignVariableOp_17AssignVariableOptraining/Adam/Variable_4Placeholder_17*
dtype0
?
ReadVariableOp_17ReadVariableOptraining/Adam/Variable_4^AssignVariableOp_17*
dtype0*!
_output_shapes
:???
Y
Placeholder_18Placeholder*
shape:?*
dtype0*
_output_shapes	
:?
^
AssignVariableOp_18AssignVariableOptraining/Adam/Variable_5Placeholder_18*
dtype0
}
ReadVariableOp_18ReadVariableOptraining/Adam/Variable_5^AssignVariableOp_18*
dtype0*
_output_shapes	
:?
c
Placeholder_19Placeholder*
dtype0* 
_output_shapes
:
??*
shape:
??
^
AssignVariableOp_19AssignVariableOptraining/Adam/Variable_6Placeholder_19*
dtype0
?
ReadVariableOp_19ReadVariableOptraining/Adam/Variable_6^AssignVariableOp_19*
dtype0* 
_output_shapes
:
??
Y
Placeholder_20Placeholder*
shape:?*
dtype0*
_output_shapes	
:?
^
AssignVariableOp_20AssignVariableOptraining/Adam/Variable_7Placeholder_20*
dtype0
}
ReadVariableOp_20ReadVariableOptraining/Adam/Variable_7^AssignVariableOp_20*
dtype0*
_output_shapes	
:?
a
Placeholder_21Placeholder*
dtype0*
_output_shapes
:	?*
shape:	?
^
AssignVariableOp_21AssignVariableOptraining/Adam/Variable_8Placeholder_21*
dtype0
?
ReadVariableOp_21ReadVariableOptraining/Adam/Variable_8^AssignVariableOp_21*
dtype0*
_output_shapes
:	?
W
Placeholder_22Placeholder*
dtype0*
_output_shapes
:*
shape:
^
AssignVariableOp_22AssignVariableOptraining/Adam/Variable_9Placeholder_22*
dtype0
|
ReadVariableOp_22ReadVariableOptraining/Adam/Variable_9^AssignVariableOp_22*
dtype0*
_output_shapes
:
a
Placeholder_23Placeholder*
shape:	?*
dtype0*
_output_shapes
:	?
_
AssignVariableOp_23AssignVariableOptraining/Adam/Variable_10Placeholder_23*
dtype0
?
ReadVariableOp_23ReadVariableOptraining/Adam/Variable_10^AssignVariableOp_23*
dtype0*
_output_shapes
:	?
W
Placeholder_24Placeholder*
dtype0*
_output_shapes
:*
shape:
_
AssignVariableOp_24AssignVariableOptraining/Adam/Variable_11Placeholder_24*
dtype0
}
ReadVariableOp_24ReadVariableOptraining/Adam/Variable_11^AssignVariableOp_24*
dtype0*
_output_shapes
:
o
Placeholder_25Placeholder*
dtype0*&
_output_shapes
: *
shape: 
_
AssignVariableOp_25AssignVariableOptraining/Adam/Variable_12Placeholder_25*
dtype0
?
ReadVariableOp_25ReadVariableOptraining/Adam/Variable_12^AssignVariableOp_25*
dtype0*&
_output_shapes
: 
W
Placeholder_26Placeholder*
shape: *
dtype0*
_output_shapes
: 
_
AssignVariableOp_26AssignVariableOptraining/Adam/Variable_13Placeholder_26*
dtype0
}
ReadVariableOp_26ReadVariableOptraining/Adam/Variable_13^AssignVariableOp_26*
dtype0*
_output_shapes
: 
o
Placeholder_27Placeholder*
shape:  *
dtype0*&
_output_shapes
:  
_
AssignVariableOp_27AssignVariableOptraining/Adam/Variable_14Placeholder_27*
dtype0
?
ReadVariableOp_27ReadVariableOptraining/Adam/Variable_14^AssignVariableOp_27*
dtype0*&
_output_shapes
:  
W
Placeholder_28Placeholder*
dtype0*
_output_shapes
: *
shape: 
_
AssignVariableOp_28AssignVariableOptraining/Adam/Variable_15Placeholder_28*
dtype0
}
ReadVariableOp_28ReadVariableOptraining/Adam/Variable_15^AssignVariableOp_28*
dtype0*
_output_shapes
: 
e
Placeholder_29Placeholder*
dtype0*!
_output_shapes
:???*
shape:???
_
AssignVariableOp_29AssignVariableOptraining/Adam/Variable_16Placeholder_29*
dtype0
?
ReadVariableOp_29ReadVariableOptraining/Adam/Variable_16^AssignVariableOp_29*
dtype0*!
_output_shapes
:???
Y
Placeholder_30Placeholder*
dtype0*
_output_shapes	
:?*
shape:?
_
AssignVariableOp_30AssignVariableOptraining/Adam/Variable_17Placeholder_30*
dtype0
~
ReadVariableOp_30ReadVariableOptraining/Adam/Variable_17^AssignVariableOp_30*
dtype0*
_output_shapes	
:?
c
Placeholder_31Placeholder*
dtype0* 
_output_shapes
:
??*
shape:
??
_
AssignVariableOp_31AssignVariableOptraining/Adam/Variable_18Placeholder_31*
dtype0
?
ReadVariableOp_31ReadVariableOptraining/Adam/Variable_18^AssignVariableOp_31*
dtype0* 
_output_shapes
:
??
Y
Placeholder_32Placeholder*
dtype0*
_output_shapes	
:?*
shape:?
_
AssignVariableOp_32AssignVariableOptraining/Adam/Variable_19Placeholder_32*
dtype0
~
ReadVariableOp_32ReadVariableOptraining/Adam/Variable_19^AssignVariableOp_32*
dtype0*
_output_shapes	
:?
a
Placeholder_33Placeholder*
shape:	?*
dtype0*
_output_shapes
:	?
_
AssignVariableOp_33AssignVariableOptraining/Adam/Variable_20Placeholder_33*
dtype0
?
ReadVariableOp_33ReadVariableOptraining/Adam/Variable_20^AssignVariableOp_33*
dtype0*
_output_shapes
:	?
W
Placeholder_34Placeholder*
dtype0*
_output_shapes
:*
shape:
_
AssignVariableOp_34AssignVariableOptraining/Adam/Variable_21Placeholder_34*
dtype0
}
ReadVariableOp_34ReadVariableOptraining/Adam/Variable_21^AssignVariableOp_34*
dtype0*
_output_shapes
:
a
Placeholder_35Placeholder*
dtype0*
_output_shapes
:	?*
shape:	?
_
AssignVariableOp_35AssignVariableOptraining/Adam/Variable_22Placeholder_35*
dtype0
?
ReadVariableOp_35ReadVariableOptraining/Adam/Variable_22^AssignVariableOp_35*
dtype0*
_output_shapes
:	?
W
Placeholder_36Placeholder*
dtype0*
_output_shapes
:*
shape:
_
AssignVariableOp_36AssignVariableOptraining/Adam/Variable_23Placeholder_36*
dtype0
}
ReadVariableOp_36ReadVariableOptraining/Adam/Variable_23^AssignVariableOp_36*
dtype0*
_output_shapes
:
W
Placeholder_37Placeholder*
dtype0*
_output_shapes
:*
shape:
_
AssignVariableOp_37AssignVariableOptraining/Adam/Variable_24Placeholder_37*
dtype0
}
ReadVariableOp_37ReadVariableOptraining/Adam/Variable_24^AssignVariableOp_37*
dtype0*
_output_shapes
:
W
Placeholder_38Placeholder*
dtype0*
_output_shapes
:*
shape:
_
AssignVariableOp_38AssignVariableOptraining/Adam/Variable_25Placeholder_38*
dtype0
}
ReadVariableOp_38ReadVariableOptraining/Adam/Variable_25^AssignVariableOp_38*
dtype0*
_output_shapes
:
W
Placeholder_39Placeholder*
dtype0*
_output_shapes
:*
shape:
_
AssignVariableOp_39AssignVariableOptraining/Adam/Variable_26Placeholder_39*
dtype0
}
ReadVariableOp_39ReadVariableOptraining/Adam/Variable_26^AssignVariableOp_39*
dtype0*
_output_shapes
:
W
Placeholder_40Placeholder*
dtype0*
_output_shapes
:*
shape:
_
AssignVariableOp_40AssignVariableOptraining/Adam/Variable_27Placeholder_40*
dtype0
}
ReadVariableOp_40ReadVariableOptraining/Adam/Variable_27^AssignVariableOp_40*
dtype0*
_output_shapes
:
W
Placeholder_41Placeholder*
dtype0*
_output_shapes
:*
shape:
_
AssignVariableOp_41AssignVariableOptraining/Adam/Variable_28Placeholder_41*
dtype0
}
ReadVariableOp_41ReadVariableOptraining/Adam/Variable_28^AssignVariableOp_41*
dtype0*
_output_shapes
:
W
Placeholder_42Placeholder*
shape:*
dtype0*
_output_shapes
:
_
AssignVariableOp_42AssignVariableOptraining/Adam/Variable_29Placeholder_42*
dtype0
}
ReadVariableOp_42ReadVariableOptraining/Adam/Variable_29^AssignVariableOp_42*
dtype0*
_output_shapes
:
W
Placeholder_43Placeholder*
dtype0*
_output_shapes
:*
shape:
_
AssignVariableOp_43AssignVariableOptraining/Adam/Variable_30Placeholder_43*
dtype0
}
ReadVariableOp_43ReadVariableOptraining/Adam/Variable_30^AssignVariableOp_43*
dtype0*
_output_shapes
:
W
Placeholder_44Placeholder*
dtype0*
_output_shapes
:*
shape:
_
AssignVariableOp_44AssignVariableOptraining/Adam/Variable_31Placeholder_44*
dtype0
}
ReadVariableOp_44ReadVariableOptraining/Adam/Variable_31^AssignVariableOp_44*
dtype0*
_output_shapes
:
W
Placeholder_45Placeholder*
dtype0*
_output_shapes
:*
shape:
_
AssignVariableOp_45AssignVariableOptraining/Adam/Variable_32Placeholder_45*
dtype0
}
ReadVariableOp_45ReadVariableOptraining/Adam/Variable_32^AssignVariableOp_45*
dtype0*
_output_shapes
:
W
Placeholder_46Placeholder*
dtype0*
_output_shapes
:*
shape:
_
AssignVariableOp_46AssignVariableOptraining/Adam/Variable_33Placeholder_46*
dtype0
}
ReadVariableOp_46ReadVariableOptraining/Adam/Variable_33^AssignVariableOp_46*
dtype0*
_output_shapes
:
W
Placeholder_47Placeholder*
dtype0*
_output_shapes
:*
shape:
_
AssignVariableOp_47AssignVariableOptraining/Adam/Variable_34Placeholder_47*
dtype0
}
ReadVariableOp_47ReadVariableOptraining/Adam/Variable_34^AssignVariableOp_47*
dtype0*
_output_shapes
:
W
Placeholder_48Placeholder*
dtype0*
_output_shapes
:*
shape:
_
AssignVariableOp_48AssignVariableOptraining/Adam/Variable_35Placeholder_48*
dtype0
}
ReadVariableOp_48ReadVariableOptraining/Adam/Variable_35^AssignVariableOp_48*
dtype0*
_output_shapes
:
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
?
save/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_f99924c1e5194703a552eb2456fd6215/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
?

save/SaveV2/tensor_namesConst*?	
value?	B?	5BAdam/beta_1BAdam/beta_2B
Adam/decayBAdam/iterationsBAdam/lrBconv2d/biasBconv2d/kernelBconv2d_1/biasBconv2d_1/kernelB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBground/biasBground/kernelBtraining/Adam/VariableBtraining/Adam/Variable_1Btraining/Adam/Variable_10Btraining/Adam/Variable_11Btraining/Adam/Variable_12Btraining/Adam/Variable_13Btraining/Adam/Variable_14Btraining/Adam/Variable_15Btraining/Adam/Variable_16Btraining/Adam/Variable_17Btraining/Adam/Variable_18Btraining/Adam/Variable_19Btraining/Adam/Variable_2Btraining/Adam/Variable_20Btraining/Adam/Variable_21Btraining/Adam/Variable_22Btraining/Adam/Variable_23Btraining/Adam/Variable_24Btraining/Adam/Variable_25Btraining/Adam/Variable_26Btraining/Adam/Variable_27Btraining/Adam/Variable_28Btraining/Adam/Variable_29Btraining/Adam/Variable_3Btraining/Adam/Variable_30Btraining/Adam/Variable_31Btraining/Adam/Variable_32Btraining/Adam/Variable_33Btraining/Adam/Variable_34Btraining/Adam/Variable_35Btraining/Adam/Variable_4Btraining/Adam/Variable_5Btraining/Adam/Variable_6Btraining/Adam/Variable_7Btraining/Adam/Variable_8Btraining/Adam/Variable_9Bweather/biasBweather/kernel*
dtype0*
_output_shapes
:5
?
save/SaveV2/shape_and_slicesConst*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:5
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp#Adam/iterations/Read/ReadVariableOpAdam/lr/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOpground/bias/Read/ReadVariableOp!ground/kernel/Read/ReadVariableOp*training/Adam/Variable/Read/ReadVariableOp,training/Adam/Variable_1/Read/ReadVariableOp-training/Adam/Variable_10/Read/ReadVariableOp-training/Adam/Variable_11/Read/ReadVariableOp-training/Adam/Variable_12/Read/ReadVariableOp-training/Adam/Variable_13/Read/ReadVariableOp-training/Adam/Variable_14/Read/ReadVariableOp-training/Adam/Variable_15/Read/ReadVariableOp-training/Adam/Variable_16/Read/ReadVariableOp-training/Adam/Variable_17/Read/ReadVariableOp-training/Adam/Variable_18/Read/ReadVariableOp-training/Adam/Variable_19/Read/ReadVariableOp,training/Adam/Variable_2/Read/ReadVariableOp-training/Adam/Variable_20/Read/ReadVariableOp-training/Adam/Variable_21/Read/ReadVariableOp-training/Adam/Variable_22/Read/ReadVariableOp-training/Adam/Variable_23/Read/ReadVariableOp-training/Adam/Variable_24/Read/ReadVariableOp-training/Adam/Variable_25/Read/ReadVariableOp-training/Adam/Variable_26/Read/ReadVariableOp-training/Adam/Variable_27/Read/ReadVariableOp-training/Adam/Variable_28/Read/ReadVariableOp-training/Adam/Variable_29/Read/ReadVariableOp,training/Adam/Variable_3/Read/ReadVariableOp-training/Adam/Variable_30/Read/ReadVariableOp-training/Adam/Variable_31/Read/ReadVariableOp-training/Adam/Variable_32/Read/ReadVariableOp-training/Adam/Variable_33/Read/ReadVariableOp-training/Adam/Variable_34/Read/ReadVariableOp-training/Adam/Variable_35/Read/ReadVariableOp,training/Adam/Variable_4/Read/ReadVariableOp,training/Adam/Variable_5/Read/ReadVariableOp,training/Adam/Variable_6/Read/ReadVariableOp,training/Adam/Variable_7/Read/ReadVariableOp,training/Adam/Variable_8/Read/ReadVariableOp,training/Adam/Variable_9/Read/ReadVariableOp weather/bias/Read/ReadVariableOp"weather/kernel/Read/ReadVariableOp*C
dtypes9
725	
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*
_output_shapes
:*
T0*

axis 
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
?

save/RestoreV2/tensor_namesConst*?	
value?	B?	5BAdam/beta_1BAdam/beta_2B
Adam/decayBAdam/iterationsBAdam/lrBconv2d/biasBconv2d/kernelBconv2d_1/biasBconv2d_1/kernelB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBground/biasBground/kernelBtraining/Adam/VariableBtraining/Adam/Variable_1Btraining/Adam/Variable_10Btraining/Adam/Variable_11Btraining/Adam/Variable_12Btraining/Adam/Variable_13Btraining/Adam/Variable_14Btraining/Adam/Variable_15Btraining/Adam/Variable_16Btraining/Adam/Variable_17Btraining/Adam/Variable_18Btraining/Adam/Variable_19Btraining/Adam/Variable_2Btraining/Adam/Variable_20Btraining/Adam/Variable_21Btraining/Adam/Variable_22Btraining/Adam/Variable_23Btraining/Adam/Variable_24Btraining/Adam/Variable_25Btraining/Adam/Variable_26Btraining/Adam/Variable_27Btraining/Adam/Variable_28Btraining/Adam/Variable_29Btraining/Adam/Variable_3Btraining/Adam/Variable_30Btraining/Adam/Variable_31Btraining/Adam/Variable_32Btraining/Adam/Variable_33Btraining/Adam/Variable_34Btraining/Adam/Variable_35Btraining/Adam/Variable_4Btraining/Adam/Variable_5Btraining/Adam/Variable_6Btraining/Adam/Variable_7Btraining/Adam/Variable_8Btraining/Adam/Variable_9Bweather/biasBweather/kernel*
dtype0*
_output_shapes
:5
?
save/RestoreV2/shape_and_slicesConst*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:5
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::*C
dtypes9
725	
N
save/Identity_1Identitysave/RestoreV2*
_output_shapes
:*
T0
T
save/AssignVariableOpAssignVariableOpAdam/beta_1save/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
V
save/AssignVariableOp_1AssignVariableOpAdam/beta_2save/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
U
save/AssignVariableOp_2AssignVariableOp
Adam/decaysave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0	*
_output_shapes
:
Z
save/AssignVariableOp_3AssignVariableOpAdam/iterationssave/Identity_4*
dtype0	
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
R
save/AssignVariableOp_4AssignVariableOpAdam/lrsave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
_output_shapes
:*
T0
V
save/AssignVariableOp_5AssignVariableOpconv2d/biassave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
T0*
_output_shapes
:
X
save/AssignVariableOp_6AssignVariableOpconv2d/kernelsave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
T0*
_output_shapes
:
X
save/AssignVariableOp_7AssignVariableOpconv2d_1/biassave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
T0*
_output_shapes
:
Z
save/AssignVariableOp_8AssignVariableOpconv2d_1/kernelsave/Identity_9*
dtype0
Q
save/Identity_10Identitysave/RestoreV2:9*
T0*
_output_shapes
:
V
save/AssignVariableOp_9AssignVariableOp
dense/biassave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:10*
_output_shapes
:*
T0
Y
save/AssignVariableOp_10AssignVariableOpdense/kernelsave/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:11*
T0*
_output_shapes
:
Y
save/AssignVariableOp_11AssignVariableOpdense_1/biassave/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:12*
T0*
_output_shapes
:
[
save/AssignVariableOp_12AssignVariableOpdense_1/kernelsave/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:13*
T0*
_output_shapes
:
X
save/AssignVariableOp_13AssignVariableOpground/biassave/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:14*
T0*
_output_shapes
:
Z
save/AssignVariableOp_14AssignVariableOpground/kernelsave/Identity_15*
dtype0
R
save/Identity_16Identitysave/RestoreV2:15*
T0*
_output_shapes
:
c
save/AssignVariableOp_15AssignVariableOptraining/Adam/Variablesave/Identity_16*
dtype0
R
save/Identity_17Identitysave/RestoreV2:16*
T0*
_output_shapes
:
e
save/AssignVariableOp_16AssignVariableOptraining/Adam/Variable_1save/Identity_17*
dtype0
R
save/Identity_18Identitysave/RestoreV2:17*
_output_shapes
:*
T0
f
save/AssignVariableOp_17AssignVariableOptraining/Adam/Variable_10save/Identity_18*
dtype0
R
save/Identity_19Identitysave/RestoreV2:18*
T0*
_output_shapes
:
f
save/AssignVariableOp_18AssignVariableOptraining/Adam/Variable_11save/Identity_19*
dtype0
R
save/Identity_20Identitysave/RestoreV2:19*
T0*
_output_shapes
:
f
save/AssignVariableOp_19AssignVariableOptraining/Adam/Variable_12save/Identity_20*
dtype0
R
save/Identity_21Identitysave/RestoreV2:20*
T0*
_output_shapes
:
f
save/AssignVariableOp_20AssignVariableOptraining/Adam/Variable_13save/Identity_21*
dtype0
R
save/Identity_22Identitysave/RestoreV2:21*
_output_shapes
:*
T0
f
save/AssignVariableOp_21AssignVariableOptraining/Adam/Variable_14save/Identity_22*
dtype0
R
save/Identity_23Identitysave/RestoreV2:22*
T0*
_output_shapes
:
f
save/AssignVariableOp_22AssignVariableOptraining/Adam/Variable_15save/Identity_23*
dtype0
R
save/Identity_24Identitysave/RestoreV2:23*
T0*
_output_shapes
:
f
save/AssignVariableOp_23AssignVariableOptraining/Adam/Variable_16save/Identity_24*
dtype0
R
save/Identity_25Identitysave/RestoreV2:24*
T0*
_output_shapes
:
f
save/AssignVariableOp_24AssignVariableOptraining/Adam/Variable_17save/Identity_25*
dtype0
R
save/Identity_26Identitysave/RestoreV2:25*
T0*
_output_shapes
:
f
save/AssignVariableOp_25AssignVariableOptraining/Adam/Variable_18save/Identity_26*
dtype0
R
save/Identity_27Identitysave/RestoreV2:26*
_output_shapes
:*
T0
f
save/AssignVariableOp_26AssignVariableOptraining/Adam/Variable_19save/Identity_27*
dtype0
R
save/Identity_28Identitysave/RestoreV2:27*
_output_shapes
:*
T0
e
save/AssignVariableOp_27AssignVariableOptraining/Adam/Variable_2save/Identity_28*
dtype0
R
save/Identity_29Identitysave/RestoreV2:28*
T0*
_output_shapes
:
f
save/AssignVariableOp_28AssignVariableOptraining/Adam/Variable_20save/Identity_29*
dtype0
R
save/Identity_30Identitysave/RestoreV2:29*
T0*
_output_shapes
:
f
save/AssignVariableOp_29AssignVariableOptraining/Adam/Variable_21save/Identity_30*
dtype0
R
save/Identity_31Identitysave/RestoreV2:30*
T0*
_output_shapes
:
f
save/AssignVariableOp_30AssignVariableOptraining/Adam/Variable_22save/Identity_31*
dtype0
R
save/Identity_32Identitysave/RestoreV2:31*
T0*
_output_shapes
:
f
save/AssignVariableOp_31AssignVariableOptraining/Adam/Variable_23save/Identity_32*
dtype0
R
save/Identity_33Identitysave/RestoreV2:32*
T0*
_output_shapes
:
f
save/AssignVariableOp_32AssignVariableOptraining/Adam/Variable_24save/Identity_33*
dtype0
R
save/Identity_34Identitysave/RestoreV2:33*
T0*
_output_shapes
:
f
save/AssignVariableOp_33AssignVariableOptraining/Adam/Variable_25save/Identity_34*
dtype0
R
save/Identity_35Identitysave/RestoreV2:34*
T0*
_output_shapes
:
f
save/AssignVariableOp_34AssignVariableOptraining/Adam/Variable_26save/Identity_35*
dtype0
R
save/Identity_36Identitysave/RestoreV2:35*
T0*
_output_shapes
:
f
save/AssignVariableOp_35AssignVariableOptraining/Adam/Variable_27save/Identity_36*
dtype0
R
save/Identity_37Identitysave/RestoreV2:36*
T0*
_output_shapes
:
f
save/AssignVariableOp_36AssignVariableOptraining/Adam/Variable_28save/Identity_37*
dtype0
R
save/Identity_38Identitysave/RestoreV2:37*
T0*
_output_shapes
:
f
save/AssignVariableOp_37AssignVariableOptraining/Adam/Variable_29save/Identity_38*
dtype0
R
save/Identity_39Identitysave/RestoreV2:38*
T0*
_output_shapes
:
e
save/AssignVariableOp_38AssignVariableOptraining/Adam/Variable_3save/Identity_39*
dtype0
R
save/Identity_40Identitysave/RestoreV2:39*
T0*
_output_shapes
:
f
save/AssignVariableOp_39AssignVariableOptraining/Adam/Variable_30save/Identity_40*
dtype0
R
save/Identity_41Identitysave/RestoreV2:40*
_output_shapes
:*
T0
f
save/AssignVariableOp_40AssignVariableOptraining/Adam/Variable_31save/Identity_41*
dtype0
R
save/Identity_42Identitysave/RestoreV2:41*
T0*
_output_shapes
:
f
save/AssignVariableOp_41AssignVariableOptraining/Adam/Variable_32save/Identity_42*
dtype0
R
save/Identity_43Identitysave/RestoreV2:42*
_output_shapes
:*
T0
f
save/AssignVariableOp_42AssignVariableOptraining/Adam/Variable_33save/Identity_43*
dtype0
R
save/Identity_44Identitysave/RestoreV2:43*
T0*
_output_shapes
:
f
save/AssignVariableOp_43AssignVariableOptraining/Adam/Variable_34save/Identity_44*
dtype0
R
save/Identity_45Identitysave/RestoreV2:44*
T0*
_output_shapes
:
f
save/AssignVariableOp_44AssignVariableOptraining/Adam/Variable_35save/Identity_45*
dtype0
R
save/Identity_46Identitysave/RestoreV2:45*
T0*
_output_shapes
:
e
save/AssignVariableOp_45AssignVariableOptraining/Adam/Variable_4save/Identity_46*
dtype0
R
save/Identity_47Identitysave/RestoreV2:46*
T0*
_output_shapes
:
e
save/AssignVariableOp_46AssignVariableOptraining/Adam/Variable_5save/Identity_47*
dtype0
R
save/Identity_48Identitysave/RestoreV2:47*
_output_shapes
:*
T0
e
save/AssignVariableOp_47AssignVariableOptraining/Adam/Variable_6save/Identity_48*
dtype0
R
save/Identity_49Identitysave/RestoreV2:48*
_output_shapes
:*
T0
e
save/AssignVariableOp_48AssignVariableOptraining/Adam/Variable_7save/Identity_49*
dtype0
R
save/Identity_50Identitysave/RestoreV2:49*
T0*
_output_shapes
:
e
save/AssignVariableOp_49AssignVariableOptraining/Adam/Variable_8save/Identity_50*
dtype0
R
save/Identity_51Identitysave/RestoreV2:50*
_output_shapes
:*
T0
e
save/AssignVariableOp_50AssignVariableOptraining/Adam/Variable_9save/Identity_51*
dtype0
R
save/Identity_52Identitysave/RestoreV2:51*
T0*
_output_shapes
:
Y
save/AssignVariableOp_51AssignVariableOpweather/biassave/Identity_52*
dtype0
R
save/Identity_53Identitysave/RestoreV2:52*
T0*
_output_shapes
:
[
save/AssignVariableOp_52AssignVariableOpweather/kernelsave/Identity_53*
dtype0
?
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_26^save/AssignVariableOp_27^save/AssignVariableOp_28^save/AssignVariableOp_29^save/AssignVariableOp_3^save/AssignVariableOp_30^save/AssignVariableOp_31^save/AssignVariableOp_32^save/AssignVariableOp_33^save/AssignVariableOp_34^save/AssignVariableOp_35^save/AssignVariableOp_36^save/AssignVariableOp_37^save/AssignVariableOp_38^save/AssignVariableOp_39^save/AssignVariableOp_4^save/AssignVariableOp_40^save/AssignVariableOp_41^save/AssignVariableOp_42^save/AssignVariableOp_43^save/AssignVariableOp_44^save/AssignVariableOp_45^save/AssignVariableOp_46^save/AssignVariableOp_47^save/AssignVariableOp_48^save/AssignVariableOp_49^save/AssignVariableOp_5^save/AssignVariableOp_50^save/AssignVariableOp_51^save/AssignVariableOp_52^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
-
save/restore_allNoOp^save/restore_shard "<
save/Const:0save/Identity:0save/restore_all (5 @F8"?8
trainable_variables?8?8
|
conv2d/kernel:0conv2d/kernel/Assign#conv2d/kernel/Read/ReadVariableOp:0(2*conv2d/kernel/Initializer/random_uniform:08
k
conv2d/bias:0conv2d/bias/Assign!conv2d/bias/Read/ReadVariableOp:0(2conv2d/bias/Initializer/zeros:08
?
conv2d_1/kernel:0conv2d_1/kernel/Assign%conv2d_1/kernel/Read/ReadVariableOp:0(2,conv2d_1/kernel/Initializer/random_uniform:08
s
conv2d_1/bias:0conv2d_1/bias/Assign#conv2d_1/bias/Read/ReadVariableOp:0(2!conv2d_1/bias/Initializer/zeros:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
?
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08
?
weather/kernel:0weather/kernel/Assign$weather/kernel/Read/ReadVariableOp:0(2+weather/kernel/Initializer/random_uniform:08
o
weather/bias:0weather/bias/Assign"weather/bias/Read/ReadVariableOp:0(2 weather/bias/Initializer/zeros:08
|
ground/kernel:0ground/kernel/Assign#ground/kernel/Read/ReadVariableOp:0(2*ground/kernel/Initializer/random_uniform:08
k
ground/bias:0ground/bias/Assign!ground/bias/Read/ReadVariableOp:0(2ground/bias/Initializer/zeros:08
?
Adam/iterations:0Adam/iterations/Assign%Adam/iterations/Read/ReadVariableOp:0(2+Adam/iterations/Initializer/initial_value:08
c
	Adam/lr:0Adam/lr/AssignAdam/lr/Read/ReadVariableOp:0(2#Adam/lr/Initializer/initial_value:08
s
Adam/beta_1:0Adam/beta_1/Assign!Adam/beta_1/Read/ReadVariableOp:0(2'Adam/beta_1/Initializer/initial_value:08
s
Adam/beta_2:0Adam/beta_2/Assign!Adam/beta_2/Read/ReadVariableOp:0(2'Adam/beta_2/Initializer/initial_value:08
o
Adam/decay:0Adam/decay/Assign Adam/decay/Read/ReadVariableOp:0(2&Adam/decay/Initializer/initial_value:08
?
training/Adam/Variable:0training/Adam/Variable/Assign,training/Adam/Variable/Read/ReadVariableOp:0(2training/Adam/zeros:08
?
training/Adam/Variable_1:0training/Adam/Variable_1/Assign.training/Adam/Variable_1/Read/ReadVariableOp:0(2training/Adam/zeros_1:08
?
training/Adam/Variable_2:0training/Adam/Variable_2/Assign.training/Adam/Variable_2/Read/ReadVariableOp:0(2training/Adam/zeros_2:08
?
training/Adam/Variable_3:0training/Adam/Variable_3/Assign.training/Adam/Variable_3/Read/ReadVariableOp:0(2training/Adam/zeros_3:08
?
training/Adam/Variable_4:0training/Adam/Variable_4/Assign.training/Adam/Variable_4/Read/ReadVariableOp:0(2training/Adam/zeros_4:08
?
training/Adam/Variable_5:0training/Adam/Variable_5/Assign.training/Adam/Variable_5/Read/ReadVariableOp:0(2training/Adam/zeros_5:08
?
training/Adam/Variable_6:0training/Adam/Variable_6/Assign.training/Adam/Variable_6/Read/ReadVariableOp:0(2training/Adam/zeros_6:08
?
training/Adam/Variable_7:0training/Adam/Variable_7/Assign.training/Adam/Variable_7/Read/ReadVariableOp:0(2training/Adam/zeros_7:08
?
training/Adam/Variable_8:0training/Adam/Variable_8/Assign.training/Adam/Variable_8/Read/ReadVariableOp:0(2training/Adam/zeros_8:08
?
training/Adam/Variable_9:0training/Adam/Variable_9/Assign.training/Adam/Variable_9/Read/ReadVariableOp:0(2training/Adam/zeros_9:08
?
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign/training/Adam/Variable_10/Read/ReadVariableOp:0(2training/Adam/zeros_10:08
?
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign/training/Adam/Variable_11/Read/ReadVariableOp:0(2training/Adam/zeros_11:08
?
training/Adam/Variable_12:0 training/Adam/Variable_12/Assign/training/Adam/Variable_12/Read/ReadVariableOp:0(2training/Adam/zeros_12:08
?
training/Adam/Variable_13:0 training/Adam/Variable_13/Assign/training/Adam/Variable_13/Read/ReadVariableOp:0(2training/Adam/zeros_13:08
?
training/Adam/Variable_14:0 training/Adam/Variable_14/Assign/training/Adam/Variable_14/Read/ReadVariableOp:0(2training/Adam/zeros_14:08
?
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign/training/Adam/Variable_15/Read/ReadVariableOp:0(2training/Adam/zeros_15:08
?
training/Adam/Variable_16:0 training/Adam/Variable_16/Assign/training/Adam/Variable_16/Read/ReadVariableOp:0(2training/Adam/zeros_16:08
?
training/Adam/Variable_17:0 training/Adam/Variable_17/Assign/training/Adam/Variable_17/Read/ReadVariableOp:0(2training/Adam/zeros_17:08
?
training/Adam/Variable_18:0 training/Adam/Variable_18/Assign/training/Adam/Variable_18/Read/ReadVariableOp:0(2training/Adam/zeros_18:08
?
training/Adam/Variable_19:0 training/Adam/Variable_19/Assign/training/Adam/Variable_19/Read/ReadVariableOp:0(2training/Adam/zeros_19:08
?
training/Adam/Variable_20:0 training/Adam/Variable_20/Assign/training/Adam/Variable_20/Read/ReadVariableOp:0(2training/Adam/zeros_20:08
?
training/Adam/Variable_21:0 training/Adam/Variable_21/Assign/training/Adam/Variable_21/Read/ReadVariableOp:0(2training/Adam/zeros_21:08
?
training/Adam/Variable_22:0 training/Adam/Variable_22/Assign/training/Adam/Variable_22/Read/ReadVariableOp:0(2training/Adam/zeros_22:08
?
training/Adam/Variable_23:0 training/Adam/Variable_23/Assign/training/Adam/Variable_23/Read/ReadVariableOp:0(2training/Adam/zeros_23:08
?
training/Adam/Variable_24:0 training/Adam/Variable_24/Assign/training/Adam/Variable_24/Read/ReadVariableOp:0(2training/Adam/zeros_24:08
?
training/Adam/Variable_25:0 training/Adam/Variable_25/Assign/training/Adam/Variable_25/Read/ReadVariableOp:0(2training/Adam/zeros_25:08
?
training/Adam/Variable_26:0 training/Adam/Variable_26/Assign/training/Adam/Variable_26/Read/ReadVariableOp:0(2training/Adam/zeros_26:08
?
training/Adam/Variable_27:0 training/Adam/Variable_27/Assign/training/Adam/Variable_27/Read/ReadVariableOp:0(2training/Adam/zeros_27:08
?
training/Adam/Variable_28:0 training/Adam/Variable_28/Assign/training/Adam/Variable_28/Read/ReadVariableOp:0(2training/Adam/zeros_28:08
?
training/Adam/Variable_29:0 training/Adam/Variable_29/Assign/training/Adam/Variable_29/Read/ReadVariableOp:0(2training/Adam/zeros_29:08
?
training/Adam/Variable_30:0 training/Adam/Variable_30/Assign/training/Adam/Variable_30/Read/ReadVariableOp:0(2training/Adam/zeros_30:08
?
training/Adam/Variable_31:0 training/Adam/Variable_31/Assign/training/Adam/Variable_31/Read/ReadVariableOp:0(2training/Adam/zeros_31:08
?
training/Adam/Variable_32:0 training/Adam/Variable_32/Assign/training/Adam/Variable_32/Read/ReadVariableOp:0(2training/Adam/zeros_32:08
?
training/Adam/Variable_33:0 training/Adam/Variable_33/Assign/training/Adam/Variable_33/Read/ReadVariableOp:0(2training/Adam/zeros_33:08
?
training/Adam/Variable_34:0 training/Adam/Variable_34/Assign/training/Adam/Variable_34/Read/ReadVariableOp:0(2training/Adam/zeros_34:08
?
training/Adam/Variable_35:0 training/Adam/Variable_35/Assign/training/Adam/Variable_35/Read/ReadVariableOp:0(2training/Adam/zeros_35:08"?8
	variables?8?8
|
conv2d/kernel:0conv2d/kernel/Assign#conv2d/kernel/Read/ReadVariableOp:0(2*conv2d/kernel/Initializer/random_uniform:08
k
conv2d/bias:0conv2d/bias/Assign!conv2d/bias/Read/ReadVariableOp:0(2conv2d/bias/Initializer/zeros:08
?
conv2d_1/kernel:0conv2d_1/kernel/Assign%conv2d_1/kernel/Read/ReadVariableOp:0(2,conv2d_1/kernel/Initializer/random_uniform:08
s
conv2d_1/bias:0conv2d_1/bias/Assign#conv2d_1/bias/Read/ReadVariableOp:0(2!conv2d_1/bias/Initializer/zeros:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
?
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08
?
weather/kernel:0weather/kernel/Assign$weather/kernel/Read/ReadVariableOp:0(2+weather/kernel/Initializer/random_uniform:08
o
weather/bias:0weather/bias/Assign"weather/bias/Read/ReadVariableOp:0(2 weather/bias/Initializer/zeros:08
|
ground/kernel:0ground/kernel/Assign#ground/kernel/Read/ReadVariableOp:0(2*ground/kernel/Initializer/random_uniform:08
k
ground/bias:0ground/bias/Assign!ground/bias/Read/ReadVariableOp:0(2ground/bias/Initializer/zeros:08
?
Adam/iterations:0Adam/iterations/Assign%Adam/iterations/Read/ReadVariableOp:0(2+Adam/iterations/Initializer/initial_value:08
c
	Adam/lr:0Adam/lr/AssignAdam/lr/Read/ReadVariableOp:0(2#Adam/lr/Initializer/initial_value:08
s
Adam/beta_1:0Adam/beta_1/Assign!Adam/beta_1/Read/ReadVariableOp:0(2'Adam/beta_1/Initializer/initial_value:08
s
Adam/beta_2:0Adam/beta_2/Assign!Adam/beta_2/Read/ReadVariableOp:0(2'Adam/beta_2/Initializer/initial_value:08
o
Adam/decay:0Adam/decay/Assign Adam/decay/Read/ReadVariableOp:0(2&Adam/decay/Initializer/initial_value:08
?
training/Adam/Variable:0training/Adam/Variable/Assign,training/Adam/Variable/Read/ReadVariableOp:0(2training/Adam/zeros:08
?
training/Adam/Variable_1:0training/Adam/Variable_1/Assign.training/Adam/Variable_1/Read/ReadVariableOp:0(2training/Adam/zeros_1:08
?
training/Adam/Variable_2:0training/Adam/Variable_2/Assign.training/Adam/Variable_2/Read/ReadVariableOp:0(2training/Adam/zeros_2:08
?
training/Adam/Variable_3:0training/Adam/Variable_3/Assign.training/Adam/Variable_3/Read/ReadVariableOp:0(2training/Adam/zeros_3:08
?
training/Adam/Variable_4:0training/Adam/Variable_4/Assign.training/Adam/Variable_4/Read/ReadVariableOp:0(2training/Adam/zeros_4:08
?
training/Adam/Variable_5:0training/Adam/Variable_5/Assign.training/Adam/Variable_5/Read/ReadVariableOp:0(2training/Adam/zeros_5:08
?
training/Adam/Variable_6:0training/Adam/Variable_6/Assign.training/Adam/Variable_6/Read/ReadVariableOp:0(2training/Adam/zeros_6:08
?
training/Adam/Variable_7:0training/Adam/Variable_7/Assign.training/Adam/Variable_7/Read/ReadVariableOp:0(2training/Adam/zeros_7:08
?
training/Adam/Variable_8:0training/Adam/Variable_8/Assign.training/Adam/Variable_8/Read/ReadVariableOp:0(2training/Adam/zeros_8:08
?
training/Adam/Variable_9:0training/Adam/Variable_9/Assign.training/Adam/Variable_9/Read/ReadVariableOp:0(2training/Adam/zeros_9:08
?
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign/training/Adam/Variable_10/Read/ReadVariableOp:0(2training/Adam/zeros_10:08
?
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign/training/Adam/Variable_11/Read/ReadVariableOp:0(2training/Adam/zeros_11:08
?
training/Adam/Variable_12:0 training/Adam/Variable_12/Assign/training/Adam/Variable_12/Read/ReadVariableOp:0(2training/Adam/zeros_12:08
?
training/Adam/Variable_13:0 training/Adam/Variable_13/Assign/training/Adam/Variable_13/Read/ReadVariableOp:0(2training/Adam/zeros_13:08
?
training/Adam/Variable_14:0 training/Adam/Variable_14/Assign/training/Adam/Variable_14/Read/ReadVariableOp:0(2training/Adam/zeros_14:08
?
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign/training/Adam/Variable_15/Read/ReadVariableOp:0(2training/Adam/zeros_15:08
?
training/Adam/Variable_16:0 training/Adam/Variable_16/Assign/training/Adam/Variable_16/Read/ReadVariableOp:0(2training/Adam/zeros_16:08
?
training/Adam/Variable_17:0 training/Adam/Variable_17/Assign/training/Adam/Variable_17/Read/ReadVariableOp:0(2training/Adam/zeros_17:08
?
training/Adam/Variable_18:0 training/Adam/Variable_18/Assign/training/Adam/Variable_18/Read/ReadVariableOp:0(2training/Adam/zeros_18:08
?
training/Adam/Variable_19:0 training/Adam/Variable_19/Assign/training/Adam/Variable_19/Read/ReadVariableOp:0(2training/Adam/zeros_19:08
?
training/Adam/Variable_20:0 training/Adam/Variable_20/Assign/training/Adam/Variable_20/Read/ReadVariableOp:0(2training/Adam/zeros_20:08
?
training/Adam/Variable_21:0 training/Adam/Variable_21/Assign/training/Adam/Variable_21/Read/ReadVariableOp:0(2training/Adam/zeros_21:08
?
training/Adam/Variable_22:0 training/Adam/Variable_22/Assign/training/Adam/Variable_22/Read/ReadVariableOp:0(2training/Adam/zeros_22:08
?
training/Adam/Variable_23:0 training/Adam/Variable_23/Assign/training/Adam/Variable_23/Read/ReadVariableOp:0(2training/Adam/zeros_23:08
?
training/Adam/Variable_24:0 training/Adam/Variable_24/Assign/training/Adam/Variable_24/Read/ReadVariableOp:0(2training/Adam/zeros_24:08
?
training/Adam/Variable_25:0 training/Adam/Variable_25/Assign/training/Adam/Variable_25/Read/ReadVariableOp:0(2training/Adam/zeros_25:08
?
training/Adam/Variable_26:0 training/Adam/Variable_26/Assign/training/Adam/Variable_26/Read/ReadVariableOp:0(2training/Adam/zeros_26:08
?
training/Adam/Variable_27:0 training/Adam/Variable_27/Assign/training/Adam/Variable_27/Read/ReadVariableOp:0(2training/Adam/zeros_27:08
?
training/Adam/Variable_28:0 training/Adam/Variable_28/Assign/training/Adam/Variable_28/Read/ReadVariableOp:0(2training/Adam/zeros_28:08
?
training/Adam/Variable_29:0 training/Adam/Variable_29/Assign/training/Adam/Variable_29/Read/ReadVariableOp:0(2training/Adam/zeros_29:08
?
training/Adam/Variable_30:0 training/Adam/Variable_30/Assign/training/Adam/Variable_30/Read/ReadVariableOp:0(2training/Adam/zeros_30:08
?
training/Adam/Variable_31:0 training/Adam/Variable_31/Assign/training/Adam/Variable_31/Read/ReadVariableOp:0(2training/Adam/zeros_31:08
?
training/Adam/Variable_32:0 training/Adam/Variable_32/Assign/training/Adam/Variable_32/Read/ReadVariableOp:0(2training/Adam/zeros_32:08
?
training/Adam/Variable_33:0 training/Adam/Variable_33/Assign/training/Adam/Variable_33/Read/ReadVariableOp:0(2training/Adam/zeros_33:08
?
training/Adam/Variable_34:0 training/Adam/Variable_34/Assign/training/Adam/Variable_34/Read/ReadVariableOp:0(2training/Adam/zeros_34:08
?
training/Adam/Variable_35:0 training/Adam/Variable_35/Assign/training/Adam/Variable_35/Read/ReadVariableOp:0(2training/Adam/zeros_35:08*?
serving_default?
=
input_image.
input_layer:0???????????;
ground/Sigmoid:0'
ground/Sigmoid:0?????????=
weather/Softmax:0(
weather/Softmax:0?????????tensorflow/serving/predict