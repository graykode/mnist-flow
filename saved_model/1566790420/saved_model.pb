Öô

ÁŁ
,
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	

ArgMin

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
¸
AsString

input"T

output"
Ttype:
2		
"
	precisionint˙˙˙˙˙˙˙˙˙"

scientificbool( "
shortestbool( "
widthint˙˙˙˙˙˙˙˙˙"
fillstring 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
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
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
ö
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

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
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.13.12
b'unknown'Ö	

global_step/Initializer/zerosConst*
dtype0	*
_output_shapes
: *
value	B	 R *
_class
loc:@global_step

global_step
VariableV2*
	container *
shape: *
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step
˛
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	
j
global_step/readIdentityglobal_step*
_output_shapes
: *
T0	*
_class
loc:@global_step
{
AKAKAKAKPlaceholder*$
shape:˙˙˙˙˙˙˙˙˙*
dtype0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙

)adanet/iteration_0/step/Initializer/zerosConst*
value	B	 R **
_class 
loc:@adanet/iteration_0/step*
dtype0	*
_output_shapes
: 
§
adanet/iteration_0/step
VariableV2*
shape: *
dtype0	*
_output_shapes
: *
shared_name **
_class 
loc:@adanet/iteration_0/step*
	container 
â
adanet/iteration_0/step/AssignAssignadanet/iteration_0/step)adanet/iteration_0/step/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	**
_class 
loc:@adanet/iteration_0/step

adanet/iteration_0/step/readIdentityadanet/iteration_0/step*
_output_shapes
: *
T0	**
_class 
loc:@adanet/iteration_0/step

Eadanet/iteration_0/subnetwork_t0_1_layer_dnn/input_layer/images/ShapeShapeAKAKAKAK*
_output_shapes
:*
T0*
out_type0

Sadanet/iteration_0/subnetwork_t0_1_layer_dnn/input_layer/images/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Uadanet/iteration_0/subnetwork_t0_1_layer_dnn/input_layer/images/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Uadanet/iteration_0/subnetwork_t0_1_layer_dnn/input_layer/images/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
š
Madanet/iteration_0/subnetwork_t0_1_layer_dnn/input_layer/images/strided_sliceStridedSliceEadanet/iteration_0/subnetwork_t0_1_layer_dnn/input_layer/images/ShapeSadanet/iteration_0/subnetwork_t0_1_layer_dnn/input_layer/images/strided_slice/stackUadanet/iteration_0/subnetwork_t0_1_layer_dnn/input_layer/images/strided_slice/stack_1Uadanet/iteration_0/subnetwork_t0_1_layer_dnn/input_layer/images/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0

Oadanet/iteration_0/subnetwork_t0_1_layer_dnn/input_layer/images/Reshape/shape/1Const*
value
B :*
dtype0*
_output_shapes
: 
Ż
Madanet/iteration_0/subnetwork_t0_1_layer_dnn/input_layer/images/Reshape/shapePackMadanet/iteration_0/subnetwork_t0_1_layer_dnn/input_layer/images/strided_sliceOadanet/iteration_0/subnetwork_t0_1_layer_dnn/input_layer/images/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
ě
Gadanet/iteration_0/subnetwork_t0_1_layer_dnn/input_layer/images/ReshapeReshapeAKAKAKAKMadanet/iteration_0/subnetwork_t0_1_layer_dnn/input_layer/images/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Jadanet/iteration_0/subnetwork_t0_1_layer_dnn/input_layer/concat/concat_dimConst*
value	B :*
dtype0*
_output_shapes
: 
Ç
?adanet/iteration_0/subnetwork_t0_1_layer_dnn/input_layer/concatIdentityGadanet/iteration_0/subnetwork_t0_1_layer_dnn/input_layer/images/Reshape*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ů
Zadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *L
_classB
@>loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel*
dtype0*
_output_shapes
:
ë
Xadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *SŻ˝*L
_classB
@>loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel*
dtype0*
_output_shapes
: 
ë
Xadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *SŻ=*L
_classB
@>loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel*
dtype0*
_output_shapes
: 
í
badanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniformZadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	 *

seed*
T0*L
_classB
@>loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel*
seed2

Xadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/Initializer/random_uniform/subSubXadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/Initializer/random_uniform/maxXadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/Initializer/random_uniform/min*
T0*L
_classB
@>loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel*
_output_shapes
: 

Xadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/Initializer/random_uniform/mulMulbadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/Initializer/random_uniform/RandomUniformXadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/Initializer/random_uniform/sub*
T0*L
_classB
@>loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel*
_output_shapes
:	 

Tadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/Initializer/random_uniformAddXadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/Initializer/random_uniform/mulXadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/Initializer/random_uniform/min*L
_classB
@>loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel*
_output_shapes
:	 *
T0
ý
9adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel
VariableV2*
_output_shapes
:	 *
shared_name *L
_classB
@>loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel*
	container *
shape:	 *
dtype0
ü
@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/AssignAssign9adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernelTadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*L
_classB
@>loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel*
validate_shape(*
_output_shapes
:	 
ý
>adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/readIdentity9adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel*
_output_shapes
:	 *
T0*L
_classB
@>loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel
â
Iadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias/Initializer/zerosConst*
valueB *    *J
_class@
><loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias*
dtype0*
_output_shapes
: 
ď
7adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *J
_class@
><loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias*
	container 
ć
>adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias/AssignAssign7adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/biasIadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*J
_class@
><loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias
ň
<adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias/readIdentity7adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias*
T0*J
_class@
><loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias*
_output_shapes
: 

9adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/MatMulMatMul?adanet/iteration_0/subnetwork_t0_1_layer_dnn/input_layer/concat>adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
transpose_a( *
transpose_b( 

:adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/BiasAddBiasAdd9adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/MatMul<adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
­
7adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/ReluRelu:adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
´
=adanet/iteration_0/subnetwork_t0_1_layer_dnn/dropout/IdentityIdentity7adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/Relu*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ý
\adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"    
   *N
_classD
B@loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel*
dtype0*
_output_shapes
:
ď
Zadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *Áž*N
_classD
B@loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel
ď
Zadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *Á>*N
_classD
B@loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel*
dtype0
ň
dadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform\adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/shape*

seed*
T0*N
_classD
B@loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel*
seed2*
dtype0*
_output_shapes

: 


Zadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/subSubZadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/maxZadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*N
_classD
B@loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel

Zadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/mulMuldadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/RandomUniformZadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/sub*N
_classD
B@loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel*
_output_shapes

: 
*
T0

Vadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/Initializer/random_uniformAddZadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/mulZadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/min*N
_classD
B@loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel*
_output_shapes

: 
*
T0
˙
;adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel
VariableV2*
	container *
shape
: 
*
dtype0*
_output_shapes

: 
*
shared_name *N
_classD
B@loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel

Badanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/AssignAssign;adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernelVadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/Initializer/random_uniform*N
_classD
B@loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel*
validate_shape(*
_output_shapes

: 
*
use_locking(*
T0

@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/readIdentity;adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel*
_output_shapes

: 
*
T0*N
_classD
B@loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel
ć
Kadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias/Initializer/zerosConst*
valueB
*    *L
_classB
@>loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias*
dtype0*
_output_shapes
:

ó
9adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias
VariableV2*L
_classB
@>loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias*
	container *
shape:
*
dtype0*
_output_shapes
:
*
shared_name 
î
@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias/AssignAssign9adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/biasKadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias/Initializer/zeros*
T0*L
_classB
@>loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias*
validate_shape(*
_output_shapes
:
*
use_locking(
ř
>adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias/readIdentity9adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias*
_output_shapes
:
*
T0*L
_classB
@>loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias

;adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/MatMulMatMul=adanet/iteration_0/subnetwork_t0_1_layer_dnn/dropout/Identity@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
transpose_a( *
transpose_b( *
T0

<adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/BiasAddBiasAdd;adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/MatMul>adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

x
6adanet/iteration_0/subnetwork_t0_1_layer_dnn/ToFloat/xConst*
value	B :*
dtype0*
_output_shapes
: 
´
4adanet/iteration_0/subnetwork_t0_1_layer_dnn/ToFloatCast6adanet/iteration_0/subnetwork_t0_1_layer_dnn/ToFloat/x*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0

1adanet/iteration_0/subnetwork_t0_1_layer_dnn/SqrtSqrt4adanet/iteration_0/subnetwork_t0_1_layer_dnn/ToFloat*
T0*
_output_shapes
: 
F
ConstConst*
valueB B *
dtype0*
_output_shapes
: 
H
Const_1Const*
valueB B *
dtype0*
_output_shapes
: 
t
2adanet/iteration_0/subnetwork_t0_1_layer_dnn/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
ş
>adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/logits/ShapeShape<adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/BiasAdd*
T0*
out_type0*
_output_shapes
:

Radanet/iteration_0/subnetwork_t0_1_layer_dnn/head/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 

|adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
u
madanet/iteration_0/subnetwork_t0_1_layer_dnn/head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp

Qadanet/iteration_0/subnetwork_t0_1_layer_dnn/head/predictions/class_ids/dimensionConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ż
Gadanet/iteration_0/subnetwork_t0_1_layer_dnn/head/predictions/class_idsArgMax<adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/BiasAddQadanet/iteration_0/subnetwork_t0_1_layer_dnn/head/predictions/class_ids/dimension*
T0*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0

Ladanet/iteration_0/subnetwork_t0_1_layer_dnn/head/predictions/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ť
Hadanet/iteration_0/subnetwork_t0_1_layer_dnn/head/predictions/ExpandDims
ExpandDimsGadanet/iteration_0/subnetwork_t0_1_layer_dnn/head/predictions/class_idsLadanet/iteration_0/subnetwork_t0_1_layer_dnn/head/predictions/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tdim0*
T0	
Ż
Iadanet/iteration_0/subnetwork_t0_1_layer_dnn/head/predictions/str_classesAsStringHadanet/iteration_0/subnetwork_t0_1_layer_dnn/head/predictions/ExpandDims*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	precision˙˙˙˙˙˙˙˙˙*
shortest( *
T0	*

fill *

scientific( *
width˙˙˙˙˙˙˙˙˙
Ć
Kadanet/iteration_0/subnetwork_t0_1_layer_dnn/head/predictions/probabilitiesSoftmax<adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
Â
7adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/ShapeShapeKadanet/iteration_0/subnetwork_t0_1_layer_dnn/head/predictions/probabilities*
_output_shapes
:*
T0*
out_type0

Eadanet/iteration_0/subnetwork_t0_1_layer_dnn/head/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Gadanet/iteration_0/subnetwork_t0_1_layer_dnn/head/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0

Gadanet/iteration_0/subnetwork_t0_1_layer_dnn/head/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ó
?adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/strided_sliceStridedSlice7adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/ShapeEadanet/iteration_0/subnetwork_t0_1_layer_dnn/head/strided_slice/stackGadanet/iteration_0/subnetwork_t0_1_layer_dnn/head/strided_slice/stack_1Gadanet/iteration_0/subnetwork_t0_1_layer_dnn/head/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 

=adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/range/startConst*
value	B : *
dtype0*
_output_shapes
: 

=adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/range/limitConst*
dtype0*
_output_shapes
: *
value	B :


=adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
Ľ
7adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/rangeRange=adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/range/start=adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/range/limit=adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/range/delta*
_output_shapes
:
*

Tidx0

:adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/AsStringAsString7adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/range*

fill *

scientific( *
width˙˙˙˙˙˙˙˙˙*
_output_shapes
:
*
	precision˙˙˙˙˙˙˙˙˙*
shortest( *
T0

@adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
ý
<adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/ExpandDims
ExpandDims:adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/AsString@adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/ExpandDims/dim*
T0*
_output_shapes

:
*

Tdim0

Badanet/iteration_0/subnetwork_t0_1_layer_dnn/head/Tile/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 

@adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/Tile/multiplesPack?adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/strided_sliceBadanet/iteration_0/subnetwork_t0_1_layer_dnn/head/Tile/multiples/1*
T0*

axis *
N*
_output_shapes
:

6adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/TileTile<adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/ExpandDims@adanet/iteration_0/subnetwork_t0_1_layer_dnn/head/Tile/multiples*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

ß
badanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/ShapeShape=adanet/iteration_0/subnetwork_t0_1_layer_dnn/dropout/Identity*
T0*
out_type0*
_output_shapes
:
ş
padanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
ź
radanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
ź
radanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ę
jadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_sliceStridedSlicebadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/Shapepadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_slice/stackradanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_slice/stack_1radanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
Ň
adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/Initializer/ConstConst*
valueB
 *  ?*
_class{
ywloc:@adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight*
dtype0*
_output_shapes
: 
Ţ
radanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class{
ywloc:@adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight
Đ
yadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/AssignAssignradanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weightadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/Initializer/Const*
_class{
ywloc:@adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
 
wadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/readIdentityradanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight*
_output_shapes
: *
T0*
_class{
ywloc:@adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight
×
gadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/MulMul<adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/BiasAddwadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


]adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/bias/Initializer/zerosConst*
valueB
*    *^
_classT
RPloc:@adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/bias*
dtype0*
_output_shapes
:


Kadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/bias
VariableV2*
shape:
*
dtype0*
_output_shapes
:
*
shared_name *^
_classT
RPloc:@adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/bias*
	container 
ś
Radanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/bias/AssignAssignKadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/bias]adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/bias/Initializer/zeros*
use_locking(*
T0*^
_classT
RPloc:@adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/bias*
validate_shape(*
_output_shapes
:

Ž
Padanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/bias/readIdentityKadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/bias*^
_classT
RPloc:@adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/bias*
_output_shapes
:
*
T0
Ĺ
Qadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/logits/AddAddPadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/bias/readgadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

đ
Oadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/norm/AbsAbswadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/read*
T0*
_output_shapes
: 

Qadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/norm/ConstConst*
valueB *
dtype0*
_output_shapes
: 
¸
Oadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/norm/SumSumOadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/norm/AbsQadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/norm/Const*
	keep_dims(*

Tidx0*
T0*
_output_shapes
: 
ä
Sadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/norm/SqueezeSqueezeOadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/norm/Sum*
T0*
_output_shapes
: *
squeeze_dims
 

Ladanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *    

Jadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/addAddLadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/add/xSadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/norm/Squeeze*
T0*
_output_shapes
: 

Kadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 

Nadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/add_1/xConst*
_output_shapes
: *
valueB
 *    *
dtype0

Ladanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/add_1AddNadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/add_1/xKadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/zero*
_output_shapes
: *
T0

Ladanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/ConstConst*
valueB B *
dtype0*
_output_shapes
: 

Nadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/Const_1Const*
valueB B *
dtype0*
_output_shapes
: 

Nadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/Const_2Const*
valueB B *
dtype0*
_output_shapes
: 

Nadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/truedivRealDivSadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/norm/SqueezeJadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/add*
T0*
_output_shapes
: 

Nadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/Const_3Const*
valueB B *
dtype0*
_output_shapes
: 
é
Xadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/logits/ShapeShapeQadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/logits/Add*
_output_shapes
:*
T0*
out_type0
Ž
ladanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 

adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp

adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
ś
kadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/predictions/class_ids/dimensionConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
ř
aadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/predictions/class_idsArgMaxQadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/logits/Addkadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/predictions/class_ids/dimension*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*
T0*
output_type0	
ą
fadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/predictions/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
ů
badanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/predictions/ExpandDims
ExpandDimsaadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/predictions/class_idsfadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/predictions/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tdim0*
T0	
ă
cadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/predictions/str_classesAsStringbadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/predictions/ExpandDims*

fill *

scientific( *
width˙˙˙˙˙˙˙˙˙*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	precision˙˙˙˙˙˙˙˙˙*
shortest( *
T0	
ő
eadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/predictions/probabilitiesSoftmaxQadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/logits/Add*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
ö
Qadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/ShapeShapeeadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/predictions/probabilities*
T0*
out_type0*
_output_shapes
:
Š
_adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ť
aadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
Ť
aadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ő
Yadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/strided_sliceStridedSliceQadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/Shape_adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/strided_slice/stackaadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/strided_slice/stack_1aadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask

Wadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/range/startConst*
value	B : *
dtype0*
_output_shapes
: 

Wadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/range/limitConst*
value	B :
*
dtype0*
_output_shapes
: 

Wadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

Qadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/rangeRangeWadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/range/startWadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/range/limitWadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/range/delta*
_output_shapes
:
*

Tidx0
ś
Tadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/AsStringAsStringQadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/range*
T0*

fill *

scientific( *
width˙˙˙˙˙˙˙˙˙*
_output_shapes
:
*
	precision˙˙˙˙˙˙˙˙˙*
shortest( 

Zadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ë
Vadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/ExpandDims
ExpandDimsTadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/AsStringZadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/ExpandDims/dim*
_output_shapes

:
*

Tdim0*
T0

\adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/Tile/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ő
Zadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/Tile/multiplesPackYadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/strided_slice\adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/Tile/multiples/1*
T0*

axis *
N*
_output_shapes
:
Đ
Padanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/TileTileVadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/ExpandDimsZadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/Tile/multiples*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*

Tmultiples0*
T0
Ś
aadanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ˇ
Sadanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
Î
Zadanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss/AssignAssignSadanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_lossaadanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss/initial_value*
use_locking(*
T0*f
_class\
ZXloc:@adanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss*
validate_shape(*
_output_shapes
: 
Â
Xadanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss/readIdentitySadanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss*
T0*f
_class\
ZXloc:@adanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss*
_output_shapes
: 

Uadanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/is_training/xConst*
_output_shapes
: *
value	B : *
dtype0

Uadanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/is_training/yConst*
value
B :Ä*
dtype0*
_output_shapes
: 
Ş
Sadanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/is_trainingLessUadanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/is_training/xUadanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/is_training/y*
T0*
_output_shapes
: 
o
-adanet/iteration_0/best_candidate_index/ConstConst*
dtype0*
_output_shapes
: *
value	B : 
Y
adanet/iteration_0/ConstConst*
valueB B *
dtype0*
_output_shapes
: 
[
adanet/iteration_0/Const_1Const*
dtype0*
_output_shapes
: *
valueB B 

adanet/iteration_0/NoOpNoOp

)adanet/iteration_1/step/Initializer/zerosConst*
value	B	 R **
_class 
loc:@adanet/iteration_1/step*
dtype0	*
_output_shapes
: 
§
adanet/iteration_1/step
VariableV2*
dtype0	*
_output_shapes
: *
shared_name **
_class 
loc:@adanet/iteration_1/step*
	container *
shape: 
â
adanet/iteration_1/step/AssignAssignadanet/iteration_1/step)adanet/iteration_1/step/Initializer/zeros*
T0	**
_class 
loc:@adanet/iteration_1/step*
validate_shape(*
_output_shapes
: *
use_locking(

adanet/iteration_1/step/readIdentityadanet/iteration_1/step*
T0	**
_class 
loc:@adanet/iteration_1/step*
_output_shapes
: 
Ś
aadanet/iteration_1/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
ˇ
Sadanet/iteration_1/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
Î
Zadanet/iteration_1/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss/AssignAssignSadanet/iteration_1/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_lossaadanet/iteration_1/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss/initial_value*f
_class\
ZXloc:@adanet/iteration_1/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Â
Xadanet/iteration_1/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss/readIdentitySadanet/iteration_1/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss*
_output_shapes
: *
T0*f
_class\
ZXloc:@adanet/iteration_1/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss

Sadanet/iteration_1/candidate_t0_1_layer_dnn_grow_complexity_regularized/is_trainingConst*
dtype0
*
_output_shapes
: *
value	B
 Z 

Eadanet/iteration_1/subnetwork_t1_1_layer_dnn/input_layer/images/ShapeShapeAKAKAKAK*
T0*
out_type0*
_output_shapes
:

Sadanet/iteration_1/subnetwork_t1_1_layer_dnn/input_layer/images/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Uadanet/iteration_1/subnetwork_t1_1_layer_dnn/input_layer/images/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Uadanet/iteration_1/subnetwork_t1_1_layer_dnn/input_layer/images/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
š
Madanet/iteration_1/subnetwork_t1_1_layer_dnn/input_layer/images/strided_sliceStridedSliceEadanet/iteration_1/subnetwork_t1_1_layer_dnn/input_layer/images/ShapeSadanet/iteration_1/subnetwork_t1_1_layer_dnn/input_layer/images/strided_slice/stackUadanet/iteration_1/subnetwork_t1_1_layer_dnn/input_layer/images/strided_slice/stack_1Uadanet/iteration_1/subnetwork_t1_1_layer_dnn/input_layer/images/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0

Oadanet/iteration_1/subnetwork_t1_1_layer_dnn/input_layer/images/Reshape/shape/1Const*
_output_shapes
: *
value
B :*
dtype0
Ż
Madanet/iteration_1/subnetwork_t1_1_layer_dnn/input_layer/images/Reshape/shapePackMadanet/iteration_1/subnetwork_t1_1_layer_dnn/input_layer/images/strided_sliceOadanet/iteration_1/subnetwork_t1_1_layer_dnn/input_layer/images/Reshape/shape/1*

axis *
N*
_output_shapes
:*
T0
ě
Gadanet/iteration_1/subnetwork_t1_1_layer_dnn/input_layer/images/ReshapeReshapeAKAKAKAKMadanet/iteration_1/subnetwork_t1_1_layer_dnn/input_layer/images/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Jadanet/iteration_1/subnetwork_t1_1_layer_dnn/input_layer/concat/concat_dimConst*
value	B :*
dtype0*
_output_shapes
: 
Ç
?adanet/iteration_1/subnetwork_t1_1_layer_dnn/input_layer/concatIdentityGadanet/iteration_1/subnetwork_t1_1_layer_dnn/input_layer/images/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ů
Zadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel*
dtype0*
_output_shapes
:
ë
Xadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *SŻ˝*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel*
dtype0*
_output_shapes
: 
ë
Xadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *SŻ=*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel*
dtype0*
_output_shapes
: 
í
badanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniformZadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/Initializer/random_uniform/shape*
T0*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel*
seed2*
dtype0*
_output_shapes
:	 *

seed

Xadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/Initializer/random_uniform/subSubXadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/Initializer/random_uniform/maxXadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel

Xadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/Initializer/random_uniform/mulMulbadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/Initializer/random_uniform/RandomUniformXadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/Initializer/random_uniform/sub*
T0*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel*
_output_shapes
:	 

Tadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/Initializer/random_uniformAddXadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/Initializer/random_uniform/mulXadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	 *
T0*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel
ý
9adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel
VariableV2*
shape:	 *
dtype0*
_output_shapes
:	 *
shared_name *L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel*
	container 
ü
@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/AssignAssign9adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernelTadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel*
validate_shape(*
_output_shapes
:	 
ý
>adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/readIdentity9adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel*
T0*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel*
_output_shapes
:	 
â
Iadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias/Initializer/zerosConst*
valueB *    *J
_class@
><loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias*
dtype0*
_output_shapes
: 
ď
7adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *J
_class@
><loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias
ć
>adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias/AssignAssign7adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/biasIadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias*
validate_shape(*
_output_shapes
: 
ň
<adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias/readIdentity7adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias*
T0*J
_class@
><loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias*
_output_shapes
: 

9adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/MatMulMatMul?adanet/iteration_1/subnetwork_t1_1_layer_dnn/input_layer/concat>adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
transpose_a( *
transpose_b( *
T0

:adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/BiasAddBiasAdd9adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/MatMul<adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
­
7adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/ReluRelu:adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
´
=adanet/iteration_1/subnetwork_t1_1_layer_dnn/dropout/IdentityIdentity7adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/Relu*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0
ý
\adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"    
   *N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel*
dtype0*
_output_shapes
:
ď
Zadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *Áž*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel*
dtype0
ď
Zadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *Á>*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel*
dtype0
ň
dadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform\adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/shape*

seed*
T0*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel*
seed2*
dtype0*
_output_shapes

: 


Zadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/subSubZadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/maxZadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/min*
T0*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel*
_output_shapes
: 

Zadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/mulMuldadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/RandomUniformZadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/sub*
_output_shapes

: 
*
T0*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel

Vadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/Initializer/random_uniformAddZadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/mulZadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/Initializer/random_uniform/min*
T0*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel*
_output_shapes

: 

˙
;adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel
VariableV2*
shared_name *N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel*
	container *
shape
: 
*
dtype0*
_output_shapes

: 


Badanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/AssignAssign;adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernelVadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/Initializer/random_uniform*
use_locking(*
T0*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel*
validate_shape(*
_output_shapes

: 


@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/readIdentity;adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel*
_output_shapes

: 
*
T0*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel
ć
Kadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias/Initializer/zerosConst*
valueB
*    *L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias*
dtype0*
_output_shapes
:

ó
9adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias
VariableV2*
_output_shapes
:
*
shared_name *L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias*
	container *
shape:
*
dtype0
î
@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias/AssignAssign9adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/biasKadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias
ř
>adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias/readIdentity9adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias*
T0*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias*
_output_shapes
:


;adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/MatMulMatMul=adanet/iteration_1/subnetwork_t1_1_layer_dnn/dropout/Identity@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
transpose_a( *
transpose_b( *
T0

<adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/BiasAddBiasAdd;adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/MatMul>adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

x
6adanet/iteration_1/subnetwork_t1_1_layer_dnn/ToFloat/xConst*
value	B :*
dtype0*
_output_shapes
: 
´
4adanet/iteration_1/subnetwork_t1_1_layer_dnn/ToFloatCast6adanet/iteration_1/subnetwork_t1_1_layer_dnn/ToFloat/x*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

1adanet/iteration_1/subnetwork_t1_1_layer_dnn/SqrtSqrt4adanet/iteration_1/subnetwork_t1_1_layer_dnn/ToFloat*
T0*
_output_shapes
: 
H
Const_2Const*
valueB B *
dtype0*
_output_shapes
: 
H
Const_3Const*
valueB B *
dtype0*
_output_shapes
: 
t
2adanet/iteration_1/subnetwork_t1_1_layer_dnn/ConstConst*
_output_shapes
: *
value	B :*
dtype0
ş
>adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/logits/ShapeShape<adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/BiasAdd*
T0*
out_type0*
_output_shapes
:

Radanet/iteration_1/subnetwork_t1_1_layer_dnn/head/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 

|adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
u
madanet/iteration_1/subnetwork_t1_1_layer_dnn/head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp

Qadanet/iteration_1/subnetwork_t1_1_layer_dnn/head/predictions/class_ids/dimensionConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ż
Gadanet/iteration_1/subnetwork_t1_1_layer_dnn/head/predictions/class_idsArgMax<adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/BiasAddQadanet/iteration_1/subnetwork_t1_1_layer_dnn/head/predictions/class_ids/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ladanet/iteration_1/subnetwork_t1_1_layer_dnn/head/predictions/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
Ť
Hadanet/iteration_1/subnetwork_t1_1_layer_dnn/head/predictions/ExpandDims
ExpandDimsGadanet/iteration_1/subnetwork_t1_1_layer_dnn/head/predictions/class_idsLadanet/iteration_1/subnetwork_t1_1_layer_dnn/head/predictions/ExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
Iadanet/iteration_1/subnetwork_t1_1_layer_dnn/head/predictions/str_classesAsStringHadanet/iteration_1/subnetwork_t1_1_layer_dnn/head/predictions/ExpandDims*
T0	*

fill *

scientific( *
width˙˙˙˙˙˙˙˙˙*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shortest( *
	precision˙˙˙˙˙˙˙˙˙
Ć
Kadanet/iteration_1/subnetwork_t1_1_layer_dnn/head/predictions/probabilitiesSoftmax<adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
Â
7adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/ShapeShapeKadanet/iteration_1/subnetwork_t1_1_layer_dnn/head/predictions/probabilities*
T0*
out_type0*
_output_shapes
:

Eadanet/iteration_1/subnetwork_t1_1_layer_dnn/head/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Gadanet/iteration_1/subnetwork_t1_1_layer_dnn/head/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Gadanet/iteration_1/subnetwork_t1_1_layer_dnn/head/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ó
?adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/strided_sliceStridedSlice7adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/ShapeEadanet/iteration_1/subnetwork_t1_1_layer_dnn/head/strided_slice/stackGadanet/iteration_1/subnetwork_t1_1_layer_dnn/head/strided_slice/stack_1Gadanet/iteration_1/subnetwork_t1_1_layer_dnn/head/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 

=adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/range/startConst*
value	B : *
dtype0*
_output_shapes
: 

=adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/range/limitConst*
value	B :
*
dtype0*
_output_shapes
: 

=adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ľ
7adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/rangeRange=adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/range/start=adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/range/limit=adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/range/delta*

Tidx0*
_output_shapes
:


:adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/AsStringAsString7adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/range*
_output_shapes
:
*
shortest( *
	precision˙˙˙˙˙˙˙˙˙*
T0*

fill *

scientific( *
width˙˙˙˙˙˙˙˙˙

@adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/ExpandDims/dimConst*
_output_shapes
: *
value	B : *
dtype0
ý
<adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/ExpandDims
ExpandDims:adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/AsString@adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/ExpandDims/dim*
T0*
_output_shapes

:
*

Tdim0

Badanet/iteration_1/subnetwork_t1_1_layer_dnn/head/Tile/multiples/1Const*
dtype0*
_output_shapes
: *
value	B :

@adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/Tile/multiplesPack?adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/strided_sliceBadanet/iteration_1/subnetwork_t1_1_layer_dnn/head/Tile/multiples/1*
T0*

axis *
N*
_output_shapes
:

6adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/TileTile<adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/ExpandDims@adanet/iteration_1/subnetwork_t1_1_layer_dnn/head/Tile/multiples*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


Eadanet/iteration_1/subnetwork_t1_2_layer_dnn/input_layer/images/ShapeShapeAKAKAKAK*
T0*
out_type0*
_output_shapes
:

Sadanet/iteration_1/subnetwork_t1_2_layer_dnn/input_layer/images/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

Uadanet/iteration_1/subnetwork_t1_2_layer_dnn/input_layer/images/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

Uadanet/iteration_1/subnetwork_t1_2_layer_dnn/input_layer/images/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
š
Madanet/iteration_1/subnetwork_t1_2_layer_dnn/input_layer/images/strided_sliceStridedSliceEadanet/iteration_1/subnetwork_t1_2_layer_dnn/input_layer/images/ShapeSadanet/iteration_1/subnetwork_t1_2_layer_dnn/input_layer/images/strided_slice/stackUadanet/iteration_1/subnetwork_t1_2_layer_dnn/input_layer/images/strided_slice/stack_1Uadanet/iteration_1/subnetwork_t1_2_layer_dnn/input_layer/images/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask

Oadanet/iteration_1/subnetwork_t1_2_layer_dnn/input_layer/images/Reshape/shape/1Const*
_output_shapes
: *
value
B :*
dtype0
Ż
Madanet/iteration_1/subnetwork_t1_2_layer_dnn/input_layer/images/Reshape/shapePackMadanet/iteration_1/subnetwork_t1_2_layer_dnn/input_layer/images/strided_sliceOadanet/iteration_1/subnetwork_t1_2_layer_dnn/input_layer/images/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 
ě
Gadanet/iteration_1/subnetwork_t1_2_layer_dnn/input_layer/images/ReshapeReshapeAKAKAKAKMadanet/iteration_1/subnetwork_t1_2_layer_dnn/input_layer/images/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Jadanet/iteration_1/subnetwork_t1_2_layer_dnn/input_layer/concat/concat_dimConst*
dtype0*
_output_shapes
: *
value	B :
Ç
?adanet/iteration_1/subnetwork_t1_2_layer_dnn/input_layer/concatIdentityGadanet/iteration_1/subnetwork_t1_2_layer_dnn/input_layer/images/Reshape*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ů
Zadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel*
dtype0*
_output_shapes
:
ë
Xadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *SŻ˝*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel*
dtype0*
_output_shapes
: 
ë
Xadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *SŻ=*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel*
dtype0*
_output_shapes
: 
í
badanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniformZadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/Initializer/random_uniform/shape*
seed2*
dtype0*
_output_shapes
:	 *

seed*
T0*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel

Xadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/Initializer/random_uniform/subSubXadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/Initializer/random_uniform/maxXadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/Initializer/random_uniform/min*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel*
_output_shapes
: *
T0

Xadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/Initializer/random_uniform/mulMulbadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/Initializer/random_uniform/RandomUniformXadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/Initializer/random_uniform/sub*
T0*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel*
_output_shapes
:	 

Tadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/Initializer/random_uniformAddXadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/Initializer/random_uniform/mulXadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/Initializer/random_uniform/min*
T0*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel*
_output_shapes
:	 
ý
9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel
VariableV2*
shared_name *L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel*
	container *
shape:	 *
dtype0*
_output_shapes
:	 
ü
@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/AssignAssign9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernelTadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/Initializer/random_uniform*
_output_shapes
:	 *
use_locking(*
T0*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel*
validate_shape(
ý
>adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/readIdentity9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel*
T0*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel*
_output_shapes
:	 
â
Iadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias/Initializer/zerosConst*
_output_shapes
: *
valueB *    *J
_class@
><loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias*
dtype0
ď
7adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias
VariableV2*
_output_shapes
: *
shared_name *J
_class@
><loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias*
	container *
shape: *
dtype0
ć
>adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias/AssignAssign7adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/biasIadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias*
validate_shape(*
_output_shapes
: 
ň
<adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias/readIdentity7adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias*
T0*J
_class@
><loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias*
_output_shapes
: 

9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/MatMulMatMul?adanet/iteration_1/subnetwork_t1_2_layer_dnn/input_layer/concat>adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
transpose_a( *
transpose_b( 

:adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/BiasAddBiasAdd9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/MatMul<adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias/read*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0
­
7adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/ReluRelu:adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
´
=adanet/iteration_1/subnetwork_t1_2_layer_dnn/dropout/IdentityIdentity7adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/Relu*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0
ý
\adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"        *N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel*
dtype0
ď
Zadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *qÄž*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel*
dtype0
ď
Zadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *qÄ>*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel*
dtype0*
_output_shapes
: 
ň
dadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform\adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:  *

seed*
T0*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel*
seed2

Zadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/Initializer/random_uniform/subSubZadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/Initializer/random_uniform/maxZadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/Initializer/random_uniform/min*
T0*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel*
_output_shapes
: 

Zadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/Initializer/random_uniform/mulMuldadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/Initializer/random_uniform/RandomUniformZadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/Initializer/random_uniform/sub*
T0*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel*
_output_shapes

:  

Vadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/Initializer/random_uniformAddZadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/Initializer/random_uniform/mulZadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/Initializer/random_uniform/min*
T0*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel*
_output_shapes

:  
˙
;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel
VariableV2*
shared_name *N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel*
	container *
shape
:  *
dtype0*
_output_shapes

:  

Badanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/AssignAssign;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernelVadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/Initializer/random_uniform*
use_locking(*
T0*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel*
validate_shape(*
_output_shapes

:  

@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/readIdentity;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel*
_output_shapes

:  *
T0*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel
ć
Kadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias/Initializer/zerosConst*
valueB *    *L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias*
dtype0*
_output_shapes
: 
ó
9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias
VariableV2*
shared_name *L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
î
@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias/AssignAssign9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/biasKadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias/Initializer/zeros*
_output_shapes
: *
use_locking(*
T0*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias*
validate_shape(
ř
>adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias/readIdentity9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias*
_output_shapes
: *
T0

;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/MatMulMatMul=adanet/iteration_1/subnetwork_t1_2_layer_dnn/dropout/Identity@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
transpose_a( *
transpose_b( *
T0

<adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/BiasAddBiasAdd;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/MatMul>adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ą
9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/ReluRelu<adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
¸
?adanet/iteration_1/subnetwork_t1_2_layer_dnn/dropout_1/IdentityIdentity9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/Relu*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ý
\adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"    
   *N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel*
dtype0
ď
Zadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *Áž*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel*
dtype0*
_output_shapes
: 
ď
Zadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *Á>*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel*
dtype0*
_output_shapes
: 
ň
dadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform\adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/Initializer/random_uniform/shape*
T0*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel*
seed2*
dtype0*
_output_shapes

: 
*

seed

Zadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/Initializer/random_uniform/subSubZadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/Initializer/random_uniform/maxZadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/Initializer/random_uniform/min*
T0*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel*
_output_shapes
: 

Zadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/Initializer/random_uniform/mulMuldadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/Initializer/random_uniform/RandomUniformZadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/Initializer/random_uniform/sub*
T0*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel*
_output_shapes

: 


Vadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/Initializer/random_uniformAddZadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/Initializer/random_uniform/mulZadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes

: 
*
T0*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel
˙
;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel
VariableV2*
shape
: 
*
dtype0*
_output_shapes

: 
*
shared_name *N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel*
	container 

Badanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/AssignAssign;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernelVadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes

: 
*
use_locking(*
T0*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel

@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/readIdentity;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel*
T0*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel*
_output_shapes

: 

ć
Kadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias/Initializer/zerosConst*
valueB
*    *L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias*
dtype0*
_output_shapes
:

ó
9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias
VariableV2*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias*
	container *
shape:
*
dtype0*
_output_shapes
:
*
shared_name 
î
@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias/AssignAssign9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/biasKadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias/Initializer/zeros*
use_locking(*
T0*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias*
validate_shape(*
_output_shapes
:

ř
>adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias/readIdentity9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias*
_output_shapes
:
*
T0
 
;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/MatMulMatMul?adanet/iteration_1/subnetwork_t1_2_layer_dnn/dropout_1/Identity@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
transpose_a( 

<adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/BiasAddBiasAdd;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/MatMul>adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
x
6adanet/iteration_1/subnetwork_t1_2_layer_dnn/ToFloat/xConst*
_output_shapes
: *
value	B	 R*
dtype0	
´
4adanet/iteration_1/subnetwork_t1_2_layer_dnn/ToFloatCast6adanet/iteration_1/subnetwork_t1_2_layer_dnn/ToFloat/x*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0

1adanet/iteration_1/subnetwork_t1_2_layer_dnn/SqrtSqrt4adanet/iteration_1/subnetwork_t1_2_layer_dnn/ToFloat*
T0*
_output_shapes
: 
H
Const_4Const*
valueB B *
dtype0*
_output_shapes
: 
H
Const_5Const*
_output_shapes
: *
valueB B *
dtype0
t
2adanet/iteration_1/subnetwork_t1_2_layer_dnn/ConstConst*
_output_shapes
: *
value	B	 R*
dtype0	
ş
>adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/logits/ShapeShape<adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:

Radanet/iteration_1/subnetwork_t1_2_layer_dnn/head/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 

|adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
u
madanet/iteration_1/subnetwork_t1_2_layer_dnn/head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp

Qadanet/iteration_1/subnetwork_t1_2_layer_dnn/head/predictions/class_ids/dimensionConst*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
Ż
Gadanet/iteration_1/subnetwork_t1_2_layer_dnn/head/predictions/class_idsArgMax<adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/BiasAddQadanet/iteration_1/subnetwork_t1_2_layer_dnn/head/predictions/class_ids/dimension*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*
T0

Ladanet/iteration_1/subnetwork_t1_2_layer_dnn/head/predictions/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
Ť
Hadanet/iteration_1/subnetwork_t1_2_layer_dnn/head/predictions/ExpandDims
ExpandDimsGadanet/iteration_1/subnetwork_t1_2_layer_dnn/head/predictions/class_idsLadanet/iteration_1/subnetwork_t1_2_layer_dnn/head/predictions/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tdim0*
T0	
Ż
Iadanet/iteration_1/subnetwork_t1_2_layer_dnn/head/predictions/str_classesAsStringHadanet/iteration_1/subnetwork_t1_2_layer_dnn/head/predictions/ExpandDims*
width˙˙˙˙˙˙˙˙˙*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shortest( *
	precision˙˙˙˙˙˙˙˙˙*
T0	*

fill *

scientific( 
Ć
Kadanet/iteration_1/subnetwork_t1_2_layer_dnn/head/predictions/probabilitiesSoftmax<adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Â
7adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/ShapeShapeKadanet/iteration_1/subnetwork_t1_2_layer_dnn/head/predictions/probabilities*
T0*
out_type0*
_output_shapes
:

Eadanet/iteration_1/subnetwork_t1_2_layer_dnn/head/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Gadanet/iteration_1/subnetwork_t1_2_layer_dnn/head/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Gadanet/iteration_1/subnetwork_t1_2_layer_dnn/head/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ó
?adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/strided_sliceStridedSlice7adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/ShapeEadanet/iteration_1/subnetwork_t1_2_layer_dnn/head/strided_slice/stackGadanet/iteration_1/subnetwork_t1_2_layer_dnn/head/strided_slice/stack_1Gadanet/iteration_1/subnetwork_t1_2_layer_dnn/head/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask

=adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/range/startConst*
value	B : *
dtype0*
_output_shapes
: 

=adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/range/limitConst*
value	B :
*
dtype0*
_output_shapes
: 

=adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ľ
7adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/rangeRange=adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/range/start=adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/range/limit=adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/range/delta*
_output_shapes
:
*

Tidx0

:adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/AsStringAsString7adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/range*

scientific( *
width˙˙˙˙˙˙˙˙˙*
_output_shapes
:
*
	precision˙˙˙˙˙˙˙˙˙*
shortest( *
T0*

fill 

@adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/ExpandDims/dimConst*
_output_shapes
: *
value	B : *
dtype0
ý
<adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/ExpandDims
ExpandDims:adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/AsString@adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/ExpandDims/dim*
_output_shapes

:
*

Tdim0*
T0

Badanet/iteration_1/subnetwork_t1_2_layer_dnn/head/Tile/multiples/1Const*
_output_shapes
: *
value	B :*
dtype0

@adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/Tile/multiplesPack?adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/strided_sliceBadanet/iteration_1/subnetwork_t1_2_layer_dnn/head/Tile/multiples/1*
_output_shapes
:*
T0*

axis *
N

6adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/TileTile<adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/ExpandDims@adanet/iteration_1/subnetwork_t1_2_layer_dnn/head/Tile/multiples*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

ß
badanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/ShapeShape=adanet/iteration_0/subnetwork_t0_1_layer_dnn/dropout/Identity*
T0*
out_type0*
_output_shapes
:
ş
padanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
ź
radanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
ź
radanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ę
jadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_sliceStridedSlicebadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/Shapepadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_slice/stackradanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_slice/stack_1radanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
Ň
adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/Initializer/ConstConst*
_output_shapes
: *
valueB
 *   ?*
_class{
ywloc:@adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight*
dtype0
Ţ
radanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class{
ywloc:@adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight
Đ
yadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/AssignAssignradanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weightadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/Initializer/Const*
use_locking(*
T0*
_class{
ywloc:@adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight*
validate_shape(*
_output_shapes
: 
 
wadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/readIdentityradanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight*
T0*
_class{
ywloc:@adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight*
_output_shapes
: 
×
gadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/MulMul<adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/BiasAddwadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

ß
badanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/ShapeShape=adanet/iteration_1/subnetwork_t1_1_layer_dnn/dropout/Identity*
_output_shapes
:*
T0*
out_type0
ş
padanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
ź
radanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
ź
radanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ę
jadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/strided_sliceStridedSlicebadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/Shapepadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/strided_slice/stackradanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/strided_slice/stack_1radanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Ň
adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/Initializer/ConstConst*
valueB
 *   ?*
_class{
ywloc:@adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight*
dtype0*
_output_shapes
: 
Ţ
radanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class{
ywloc:@adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight*
	container *
shape: 
Đ
yadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/AssignAssignradanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weightadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/Initializer/Const*
use_locking(*
T0*
_class{
ywloc:@adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight*
validate_shape(*
_output_shapes
: 
 
wadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/readIdentityradanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight*
T0*
_class{
ywloc:@adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight*
_output_shapes
: 
×
gadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/MulMul<adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/BiasAddwadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0

]adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/bias/Initializer/zerosConst*
valueB
*    *^
_classT
RPloc:@adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/bias*
dtype0*
_output_shapes
:


Kadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/bias
VariableV2*
dtype0*
_output_shapes
:
*
shared_name *^
_classT
RPloc:@adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/bias*
	container *
shape:

ś
Radanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/bias/AssignAssignKadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/bias]adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*^
_classT
RPloc:@adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/bias
Ž
Padanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/bias/readIdentityKadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/bias*
T0*^
_classT
RPloc:@adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/bias*
_output_shapes
:

Ĺ
Qadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/logits/AddAddPadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/bias/readgadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Č
Sadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/logits/Add_1AddQadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/logits/Addgadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

đ
Oadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/norm/AbsAbswadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/read*
T0*
_output_shapes
: 

Qadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/norm/ConstConst*
valueB *
dtype0*
_output_shapes
: 
¸
Oadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/norm/SumSumOadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/norm/AbsQadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/norm/Const*
	keep_dims(*

Tidx0*
T0*
_output_shapes
: 
ä
Sadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/norm/SqueezeSqueezeOadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/norm/Sum*
squeeze_dims
 *
T0*
_output_shapes
: 

Ladanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *    

Jadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/addAddLadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/add/xSadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/norm/Squeeze*
T0*
_output_shapes
: 

Kadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 

Nadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/add_1/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 

Ladanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/add_1AddNadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/add_1/xKadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/zero*
T0*
_output_shapes
: 
ň
Qadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/norm_1/AbsAbswadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/read*
T0*
_output_shapes
: 

Sadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/norm_1/ConstConst*
valueB *
dtype0*
_output_shapes
: 
ž
Qadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/norm_1/SumSumQadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/norm_1/AbsSadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/norm_1/Const*
	keep_dims(*

Tidx0*
T0*
_output_shapes
: 
č
Uadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/norm_1/SqueezeSqueezeQadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/norm_1/Sum*
T0*
_output_shapes
: *
squeeze_dims
 

Ladanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/add_2AddJadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/addUadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/norm_1/Squeeze*
T0*
_output_shapes
: 

Madanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/zero_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 

Ladanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/add_3AddLadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/add_1Madanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/zero_1*
_output_shapes
: *
T0

Ladanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/ConstConst*
dtype0*
_output_shapes
: *
valueB B 

Nadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/Const_1Const*
_output_shapes
: *
valueB B *
dtype0

Nadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/Const_2Const*
_output_shapes
: *
valueB B *
dtype0

Nadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/truedivRealDivSadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/norm/SqueezeLadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/add_2*
T0*
_output_shapes
: 

Nadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/Const_3Const*
dtype0*
_output_shapes
: *
valueB B 

Nadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/Const_4Const*
valueB B *
dtype0*
_output_shapes
: 
Ą
Padanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/truediv_1RealDivUadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/norm_1/SqueezeLadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/add_2*
_output_shapes
: *
T0

Nadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/Const_5Const*
valueB B *
dtype0*
_output_shapes
: 
ë
Xadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/logits/ShapeShapeSadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/logits/Add_1*
T0*
out_type0*
_output_shapes
:
Ž
ladanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/logits/assert_rank_at_least/rankConst*
_output_shapes
: *
value	B :*
dtype0

adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp

adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
ś
kadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/predictions/class_ids/dimensionConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
ú
aadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/predictions/class_idsArgMaxSadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/logits/Add_1kadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/predictions/class_ids/dimension*
T0*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
ą
fadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/predictions/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
ů
badanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/predictions/ExpandDims
ExpandDimsaadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/predictions/class_idsfadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/predictions/ExpandDims/dim*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tdim0
ă
cadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/predictions/str_classesAsStringbadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/predictions/ExpandDims*

fill *

scientific( *
width˙˙˙˙˙˙˙˙˙*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shortest( *
	precision˙˙˙˙˙˙˙˙˙*
T0	
÷
eadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/predictions/probabilitiesSoftmaxSadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/logits/Add_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

ö
Qadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/ShapeShapeeadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/predictions/probabilities*
T0*
out_type0*
_output_shapes
:
Š
_adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ť
aadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ť
aadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ő
Yadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/strided_sliceStridedSliceQadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/Shape_adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/strided_slice/stackaadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/strided_slice/stack_1aadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0

Wadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/range/startConst*
value	B : *
dtype0*
_output_shapes
: 

Wadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/range/limitConst*
value	B :
*
dtype0*
_output_shapes
: 

Wadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

Qadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/rangeRangeWadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/range/startWadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/range/limitWadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/range/delta*
_output_shapes
:
*

Tidx0
ś
Tadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/AsStringAsStringQadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/range*

fill *

scientific( *
width˙˙˙˙˙˙˙˙˙*
_output_shapes
:
*
shortest( *
	precision˙˙˙˙˙˙˙˙˙*
T0

Zadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ë
Vadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/ExpandDims
ExpandDimsTadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/AsStringZadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/ExpandDims/dim*
_output_shapes

:
*

Tdim0*
T0

\adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/Tile/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ő
Zadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/Tile/multiplesPackYadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/strided_slice\adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/Tile/multiples/1*
T0*

axis *
N*
_output_shapes
:
Đ
Padanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/TileTileVadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/ExpandDimsZadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/Tile/multiples*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*

Tmultiples0*
T0
Ś
aadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/adanet_loss/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ˇ
Sadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/adanet_loss
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Î
Zadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/adanet_loss/AssignAssignSadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/adanet_lossaadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/adanet_loss/initial_value*
_output_shapes
: *
use_locking(*
T0*f
_class\
ZXloc:@adanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/adanet_loss*
validate_shape(
Â
Xadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/adanet_loss/readIdentitySadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/adanet_loss*
T0*f
_class\
ZXloc:@adanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/adanet_loss*
_output_shapes
: 

Uadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/is_training/xConst*
value	B : *
dtype0*
_output_shapes
: 

Uadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/is_training/yConst*
value
B :Ä*
dtype0*
_output_shapes
: 
Ş
Sadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/is_trainingLessUadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/is_training/xUadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/is_training/y*
T0*
_output_shapes
: 
ß
badanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/ShapeShape=adanet/iteration_0/subnetwork_t0_1_layer_dnn/dropout/Identity*
T0*
out_type0*
_output_shapes
:
ş
padanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
ź
radanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
ź
radanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ę
jadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_sliceStridedSlicebadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/Shapepadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_slice/stackradanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_slice/stack_1radanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/strided_slice/stack_2*
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask 
Ň
adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/Initializer/ConstConst*
valueB
 *   ?*
_class{
ywloc:@adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight*
dtype0*
_output_shapes
: 
Ţ
radanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class{
ywloc:@adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight*
	container *
shape: 
Đ
yadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/AssignAssignradanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weightadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/Initializer/Const*
_output_shapes
: *
use_locking(*
T0*
_class{
ywloc:@adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight*
validate_shape(
 
wadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/readIdentityradanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight*
T0*
_class{
ywloc:@adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight*
_output_shapes
: 
×
gadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/MulMul<adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/BiasAddwadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

á
badanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/ShapeShape?adanet/iteration_1/subnetwork_t1_2_layer_dnn/dropout_1/Identity*
out_type0*
_output_shapes
:*
T0
ş
padanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
ź
radanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
ź
radanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ę
jadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/strided_sliceStridedSlicebadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/Shapepadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/strided_slice/stackradanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/strided_slice/stack_1radanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Ň
adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/Initializer/ConstConst*
valueB
 *   ?*
_class{
ywloc:@adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight*
dtype0*
_output_shapes
: 
Ţ
radanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class{
ywloc:@adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight*
	container 
Đ
yadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/AssignAssignradanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weightadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/Initializer/Const*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class{
ywloc:@adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight
 
wadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/readIdentityradanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight*
_output_shapes
: *
T0*
_class{
ywloc:@adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight
×
gadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/MulMul<adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/BiasAddwadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


]adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:
*
valueB
*    *^
_classT
RPloc:@adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/bias

Kadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/bias
VariableV2*
shared_name *^
_classT
RPloc:@adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/bias*
	container *
shape:
*
dtype0*
_output_shapes
:

ś
Radanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/bias/AssignAssignKadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/bias]adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*^
_classT
RPloc:@adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/bias
Ž
Padanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/bias/readIdentityKadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/bias*
T0*^
_classT
RPloc:@adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/bias*
_output_shapes
:

Ĺ
Qadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/logits/AddAddPadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/bias/readgadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Č
Sadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/logits/Add_1AddQadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/logits/Addgadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

đ
Oadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/norm/AbsAbswadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/read*
T0*
_output_shapes
: 

Qadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/norm/ConstConst*
valueB *
dtype0*
_output_shapes
: 
¸
Oadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/norm/SumSumOadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/norm/AbsQadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/norm/Const*
	keep_dims(*

Tidx0*
T0*
_output_shapes
: 
ä
Sadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/norm/SqueezeSqueezeOadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/norm/Sum*
squeeze_dims
 *
T0*
_output_shapes
: 

Ladanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 

Jadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/addAddLadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/add/xSadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/norm/Squeeze*
T0*
_output_shapes
: 

Kadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 

Nadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/add_1/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 

Ladanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/add_1AddNadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/add_1/xKadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/zero*
T0*
_output_shapes
: 
ň
Qadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/norm_1/AbsAbswadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/read*
_output_shapes
: *
T0

Sadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/norm_1/ConstConst*
valueB *
dtype0*
_output_shapes
: 
ž
Qadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/norm_1/SumSumQadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/norm_1/AbsSadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/norm_1/Const*
T0*
_output_shapes
: *
	keep_dims(*

Tidx0
č
Uadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/norm_1/SqueezeSqueezeQadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/norm_1/Sum*
T0*
_output_shapes
: *
squeeze_dims
 

Ladanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/add_2AddJadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/addUadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/norm_1/Squeeze*
T0*
_output_shapes
: 

Madanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/zero_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 

Ladanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/add_3AddLadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/add_1Madanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/zero_1*
T0*
_output_shapes
: 

Ladanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/ConstConst*
_output_shapes
: *
valueB B *
dtype0

Nadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/Const_1Const*
valueB B *
dtype0*
_output_shapes
: 

Nadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/Const_2Const*
_output_shapes
: *
valueB B *
dtype0

Nadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/truedivRealDivSadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/norm/SqueezeLadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/add_2*
_output_shapes
: *
T0

Nadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/Const_3Const*
valueB B *
dtype0*
_output_shapes
: 

Nadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/Const_4Const*
valueB B *
dtype0*
_output_shapes
: 
Ą
Padanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/truediv_1RealDivUadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/norm_1/SqueezeLadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/add_2*
_output_shapes
: *
T0

Nadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/Const_5Const*
valueB B *
dtype0*
_output_shapes
: 
ë
Xadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/logits/ShapeShapeSadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/logits/Add_1*
T0*
out_type0*
_output_shapes
:
Ž
ladanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/logits/assert_rank_at_least/rankConst*
_output_shapes
: *
value	B :*
dtype0

adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp

adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
ś
kadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/predictions/class_ids/dimensionConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
ú
aadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/predictions/class_idsArgMaxSadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/logits/Add_1kadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/predictions/class_ids/dimension*
T0*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
ą
fadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/predictions/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
ů
badanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/predictions/ExpandDims
ExpandDimsaadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/predictions/class_idsfadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/predictions/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tdim0*
T0	
ă
cadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/predictions/str_classesAsStringbadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/predictions/ExpandDims*
T0	*

fill *

scientific( *
width˙˙˙˙˙˙˙˙˙*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shortest( *
	precision˙˙˙˙˙˙˙˙˙
÷
eadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/predictions/probabilitiesSoftmaxSadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/logits/Add_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
ö
Qadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/ShapeShapeeadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/predictions/probabilities*
T0*
out_type0*
_output_shapes
:
Š
_adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
Ť
aadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ť
aadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ő
Yadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/strided_sliceStridedSliceQadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/Shape_adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/strided_slice/stackaadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/strided_slice/stack_1aadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask

Wadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/range/startConst*
_output_shapes
: *
value	B : *
dtype0

Wadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/range/limitConst*
value	B :
*
dtype0*
_output_shapes
: 

Wadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

Qadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/rangeRangeWadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/range/startWadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/range/limitWadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/range/delta*
_output_shapes
:
*

Tidx0
ś
Tadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/AsStringAsStringQadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/range*
width˙˙˙˙˙˙˙˙˙*
_output_shapes
:
*
	precision˙˙˙˙˙˙˙˙˙*
shortest( *
T0*

fill *

scientific( 

Zadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ë
Vadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/ExpandDims
ExpandDimsTadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/AsStringZadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/ExpandDims/dim*
_output_shapes

:
*

Tdim0*
T0

\adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/Tile/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ő
Zadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/Tile/multiplesPackYadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/strided_slice\adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/Tile/multiples/1*
T0*

axis *
N*
_output_shapes
:
Đ
Padanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/TileTileVadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/ExpandDimsZadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/Tile/multiples*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ś
aadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/adanet_loss/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ˇ
Sadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/adanet_loss
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Î
Zadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/adanet_loss/AssignAssignSadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/adanet_lossaadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/adanet_loss/initial_value*f
_class\
ZXloc:@adanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/adanet_loss*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Â
Xadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/adanet_loss/readIdentitySadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/adanet_loss*
T0*f
_class\
ZXloc:@adanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/adanet_loss*
_output_shapes
: 

Uadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/is_training/xConst*
value	B : *
dtype0*
_output_shapes
: 

Uadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/is_training/yConst*
_output_shapes
: *
value
B :Ä*
dtype0
Ş
Sadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/is_trainingLessUadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/is_training/xUadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/is_training/y*
_output_shapes
: *
T0

4adanet/iteration_1/best_candidate_index/ArgMin/inputPackXadanet/iteration_1/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss/readXadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/adanet_loss/readXadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/adanet_loss/read*
T0*

axis *
N*
_output_shapes
:
z
8adanet/iteration_1/best_candidate_index/ArgMin/dimensionConst*
dtype0*
_output_shapes
: *
value	B : 
č
.adanet/iteration_1/best_candidate_index/ArgMinArgMin4adanet/iteration_1/best_candidate_index/ArgMin/input8adanet/iteration_1/best_candidate_index/ArgMin/dimension*
T0*
output_type0	*
_output_shapes
: *

Tidx0
¨
)adanet/iteration_1/best_predictions/stackPackbadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/predictions/ExpandDimsbadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/predictions/ExpandDimsbadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/predictions/ExpandDims*
T0	*

axis *
N*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
)adanet/iteration_1/best_predictions/add/yConst*
_output_shapes
: *
value	B	 R*
dtype0	
Ş
'adanet/iteration_1/best_predictions/addAdd.adanet/iteration_1/best_candidate_index/ArgMin)adanet/iteration_1/best_predictions/add/y*
_output_shapes
: *
T0	
Š
7adanet/iteration_1/best_predictions/strided_slice/stackPack.adanet/iteration_1/best_candidate_index/ArgMin*
T0	*

axis *
N*
_output_shapes
:
¤
9adanet/iteration_1/best_predictions/strided_slice/stack_1Pack'adanet/iteration_1/best_predictions/add*
T0	*

axis *
N*
_output_shapes
:

9adanet/iteration_1/best_predictions/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
˝
6adanet/iteration_1/best_predictions/strided_slice/CastCast9adanet/iteration_1/best_predictions/strided_slice/stack_2*
Truncate( *
_output_shapes
:*

DstT0	*

SrcT0
ť
1adanet/iteration_1/best_predictions/strided_sliceStridedSlice)adanet/iteration_1/best_predictions/stack7adanet/iteration_1/best_predictions/strided_slice/stack9adanet/iteration_1/best_predictions/strided_slice/stack_16adanet/iteration_1/best_predictions/strided_slice/Cast*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Index0	*
T0	
­
+adanet/iteration_1/best_predictions/stack_1Packcadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/predictions/str_classescadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/predictions/str_classescadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/predictions/str_classes*
T0*

axis *
N*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
m
+adanet/iteration_1/best_predictions/add_1/yConst*
dtype0	*
_output_shapes
: *
value	B	 R
Ž
)adanet/iteration_1/best_predictions/add_1Add.adanet/iteration_1/best_candidate_index/ArgMin+adanet/iteration_1/best_predictions/add_1/y*
_output_shapes
: *
T0	
Ť
9adanet/iteration_1/best_predictions/strided_slice_1/stackPack.adanet/iteration_1/best_candidate_index/ArgMin*
T0	*

axis *
N*
_output_shapes
:
¨
;adanet/iteration_1/best_predictions/strided_slice_1/stack_1Pack)adanet/iteration_1/best_predictions/add_1*
T0	*

axis *
N*
_output_shapes
:

;adanet/iteration_1/best_predictions/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Á
8adanet/iteration_1/best_predictions/strided_slice_1/CastCast;adanet/iteration_1/best_predictions/strided_slice_1/stack_2*
Truncate( *
_output_shapes
:*

DstT0	*

SrcT0
Ĺ
3adanet/iteration_1/best_predictions/strided_slice_1StridedSlice+adanet/iteration_1/best_predictions/stack_19adanet/iteration_1/best_predictions/strided_slice_1/stack;adanet/iteration_1/best_predictions/strided_slice_1/stack_18adanet/iteration_1/best_predictions/strided_slice_1/Cast*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Index0	
ű
+adanet/iteration_1/best_predictions/stack_2PackQadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/logits/AddSadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/logits/Add_1Sadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/logits/Add_1*
T0*

axis *
N*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

m
+adanet/iteration_1/best_predictions/add_2/yConst*
value	B	 R*
dtype0	*
_output_shapes
: 
Ž
)adanet/iteration_1/best_predictions/add_2Add.adanet/iteration_1/best_candidate_index/ArgMin+adanet/iteration_1/best_predictions/add_2/y*
_output_shapes
: *
T0	
Ť
9adanet/iteration_1/best_predictions/strided_slice_2/stackPack.adanet/iteration_1/best_candidate_index/ArgMin*
T0	*

axis *
N*
_output_shapes
:
¨
;adanet/iteration_1/best_predictions/strided_slice_2/stack_1Pack)adanet/iteration_1/best_predictions/add_2*
N*
_output_shapes
:*
T0	*

axis 

;adanet/iteration_1/best_predictions/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Á
8adanet/iteration_1/best_predictions/strided_slice_2/CastCast;adanet/iteration_1/best_predictions/strided_slice_2/stack_2*

SrcT0*
Truncate( *
_output_shapes
:*

DstT0	
Ĺ
3adanet/iteration_1/best_predictions/strided_slice_2StridedSlice+adanet/iteration_1/best_predictions/stack_29adanet/iteration_1/best_predictions/strided_slice_2/stack;adanet/iteration_1/best_predictions/strided_slice_2/stack_18adanet/iteration_1/best_predictions/strided_slice_2/Cast*
Index0	*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *'
_output_shapes
:˙˙˙˙˙˙˙˙˙

ł
+adanet/iteration_1/best_predictions/stack_3Packeadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/predictions/probabilitieseadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/predictions/probabilitieseadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/predictions/probabilities*
N*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*

axis 
m
+adanet/iteration_1/best_predictions/add_3/yConst*
value	B	 R*
dtype0	*
_output_shapes
: 
Ž
)adanet/iteration_1/best_predictions/add_3Add.adanet/iteration_1/best_candidate_index/ArgMin+adanet/iteration_1/best_predictions/add_3/y*
_output_shapes
: *
T0	
Ť
9adanet/iteration_1/best_predictions/strided_slice_3/stackPack.adanet/iteration_1/best_candidate_index/ArgMin*
T0	*

axis *
N*
_output_shapes
:
¨
;adanet/iteration_1/best_predictions/strided_slice_3/stack_1Pack)adanet/iteration_1/best_predictions/add_3*
T0	*

axis *
N*
_output_shapes
:

;adanet/iteration_1/best_predictions/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Á
8adanet/iteration_1/best_predictions/strided_slice_3/CastCast;adanet/iteration_1/best_predictions/strided_slice_3/stack_2*

SrcT0*
Truncate( *
_output_shapes
:*

DstT0	
Ĺ
3adanet/iteration_1/best_predictions/strided_slice_3StridedSlice+adanet/iteration_1/best_predictions/stack_39adanet/iteration_1/best_predictions/strided_slice_3/stack;adanet/iteration_1/best_predictions/strided_slice_3/stack_18adanet/iteration_1/best_predictions/strided_slice_3/Cast*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
Index0	*
T0
´
,adanet/iteration_1/best_export_outputs/stackPackeadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/predictions/probabilitieseadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/predictions/probabilitieseadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/predictions/probabilities*

axis *
N*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
n
,adanet/iteration_1/best_export_outputs/add/yConst*
_output_shapes
: *
value	B	 R*
dtype0	
°
*adanet/iteration_1/best_export_outputs/addAdd.adanet/iteration_1/best_candidate_index/ArgMin,adanet/iteration_1/best_export_outputs/add/y*
T0	*
_output_shapes
: 
Ź
:adanet/iteration_1/best_export_outputs/strided_slice/stackPack.adanet/iteration_1/best_candidate_index/ArgMin*
T0	*

axis *
N*
_output_shapes
:
Ş
<adanet/iteration_1/best_export_outputs/strided_slice/stack_1Pack*adanet/iteration_1/best_export_outputs/add*
_output_shapes
:*
T0	*

axis *
N

<adanet/iteration_1/best_export_outputs/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ă
9adanet/iteration_1/best_export_outputs/strided_slice/CastCast<adanet/iteration_1/best_export_outputs/strided_slice/stack_2*

SrcT0*
Truncate( *
_output_shapes
:*

DstT0	
Ę
4adanet/iteration_1/best_export_outputs/strided_sliceStridedSlice,adanet/iteration_1/best_export_outputs/stack:adanet/iteration_1/best_export_outputs/strided_slice/stack<adanet/iteration_1/best_export_outputs/strided_slice/stack_19adanet/iteration_1/best_export_outputs/strided_slice/Cast*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
Index0	*
T0
÷
.adanet/iteration_1/best_export_outputs/stack_1PackPadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/TilePadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/TilePadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/Tile*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*

axis *
N
p
.adanet/iteration_1/best_export_outputs/add_1/yConst*
value	B	 R*
dtype0	*
_output_shapes
: 
´
,adanet/iteration_1/best_export_outputs/add_1Add.adanet/iteration_1/best_candidate_index/ArgMin.adanet/iteration_1/best_export_outputs/add_1/y*
T0	*
_output_shapes
: 
Ž
<adanet/iteration_1/best_export_outputs/strided_slice_1/stackPack.adanet/iteration_1/best_candidate_index/ArgMin*
T0	*

axis *
N*
_output_shapes
:
Ž
>adanet/iteration_1/best_export_outputs/strided_slice_1/stack_1Pack,adanet/iteration_1/best_export_outputs/add_1*
T0	*

axis *
N*
_output_shapes
:

>adanet/iteration_1/best_export_outputs/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ç
;adanet/iteration_1/best_export_outputs/strided_slice_1/CastCast>adanet/iteration_1/best_export_outputs/strided_slice_1/stack_2*
Truncate( *
_output_shapes
:*

DstT0	*

SrcT0
Ô
6adanet/iteration_1/best_export_outputs/strided_slice_1StridedSlice.adanet/iteration_1/best_export_outputs/stack_1<adanet/iteration_1/best_export_outputs/strided_slice_1/stack>adanet/iteration_1/best_export_outputs/strided_slice_1/stack_1;adanet/iteration_1/best_export_outputs/strided_slice_1/Cast*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*
Index0	
ś
.adanet/iteration_1/best_export_outputs/stack_2Packeadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/predictions/probabilitieseadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/predictions/probabilitieseadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/predictions/probabilities*
T0*

axis *
N*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

p
.adanet/iteration_1/best_export_outputs/add_2/yConst*
value	B	 R*
dtype0	*
_output_shapes
: 
´
,adanet/iteration_1/best_export_outputs/add_2Add.adanet/iteration_1/best_candidate_index/ArgMin.adanet/iteration_1/best_export_outputs/add_2/y*
T0	*
_output_shapes
: 
Ž
<adanet/iteration_1/best_export_outputs/strided_slice_2/stackPack.adanet/iteration_1/best_candidate_index/ArgMin*
N*
_output_shapes
:*
T0	*

axis 
Ž
>adanet/iteration_1/best_export_outputs/strided_slice_2/stack_1Pack,adanet/iteration_1/best_export_outputs/add_2*
N*
_output_shapes
:*
T0	*

axis 

>adanet/iteration_1/best_export_outputs/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ç
;adanet/iteration_1/best_export_outputs/strided_slice_2/CastCast>adanet/iteration_1/best_export_outputs/strided_slice_2/stack_2*
_output_shapes
:*

DstT0	*

SrcT0*
Truncate( 
Ô
6adanet/iteration_1/best_export_outputs/strided_slice_2StridedSlice.adanet/iteration_1/best_export_outputs/stack_2<adanet/iteration_1/best_export_outputs/strided_slice_2/stack>adanet/iteration_1/best_export_outputs/strided_slice_2/stack_1;adanet/iteration_1/best_export_outputs/strided_slice_2/Cast*
new_axis_mask *
end_mask *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*
Index0	*
shrink_axis_mask*

begin_mask *
ellipsis_mask 
÷
.adanet/iteration_1/best_export_outputs/stack_3PackPadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/head/TilePadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/head/TilePadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/head/Tile*
T0*

axis *
N*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

p
.adanet/iteration_1/best_export_outputs/add_3/yConst*
dtype0	*
_output_shapes
: *
value	B	 R
´
,adanet/iteration_1/best_export_outputs/add_3Add.adanet/iteration_1/best_candidate_index/ArgMin.adanet/iteration_1/best_export_outputs/add_3/y*
T0	*
_output_shapes
: 
Ž
<adanet/iteration_1/best_export_outputs/strided_slice_3/stackPack.adanet/iteration_1/best_candidate_index/ArgMin*
T0	*

axis *
N*
_output_shapes
:
Ž
>adanet/iteration_1/best_export_outputs/strided_slice_3/stack_1Pack,adanet/iteration_1/best_export_outputs/add_3*
_output_shapes
:*
T0	*

axis *
N

>adanet/iteration_1/best_export_outputs/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ç
;adanet/iteration_1/best_export_outputs/strided_slice_3/CastCast>adanet/iteration_1/best_export_outputs/strided_slice_3/stack_2*

SrcT0*
Truncate( *
_output_shapes
:*

DstT0	
Ô
6adanet/iteration_1/best_export_outputs/strided_slice_3StridedSlice.adanet/iteration_1/best_export_outputs/stack_3<adanet/iteration_1/best_export_outputs/strided_slice_3/stack>adanet/iteration_1/best_export_outputs/strided_slice_3/stack_1;adanet/iteration_1/best_export_outputs/strided_slice_3/Cast*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*
Index0	
Y
adanet/iteration_1/ConstConst*
valueB B *
dtype0*
_output_shapes
: 
[
adanet/iteration_1/Const_1Const*
_output_shapes
: *
valueB B *
dtype0

adanet/iteration_1/NoOpNoOp

#current_iteration/Initializer/zerosConst*
value	B	 R *$
_class
loc:@current_iteration*
dtype0	*
_output_shapes
: 

current_iteration
VariableV2*
shared_name *$
_class
loc:@current_iteration*
	container *
shape: *
dtype0	*
_output_shapes
: 
Ę
current_iteration/AssignAssigncurrent_iteration#current_iteration/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*$
_class
loc:@current_iteration
|
current_iteration/readIdentitycurrent_iteration*
T0	*$
_class
loc:@current_iteration*
_output_shapes
: 
H
Const_6Const*
dtype0*
_output_shapes
: *
valueB B 

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_b4fdf90d81b54165ab43526900b1759c/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
ţ
save/SaveV2/tensor_namesConst"/device:CPU:0*˘
valueBBSadanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_lossBKadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/biasBradanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weightBadanet/iteration_0/stepB7adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/biasB9adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernelB9adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/biasB;adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernelBSadanet/iteration_1/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_lossBSadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/adanet_lossBSadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/adanet_lossBKadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/biasBradanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weightBradanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weightBKadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/biasBradanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weightBradanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weightBadanet/iteration_1/stepB7adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/biasB9adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernelB9adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/biasB;adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernelB7adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/biasB9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernelB9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/biasB;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernelB9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/biasB;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernelBcurrent_iterationBglobal_step*
dtype0*
_output_shapes
:
Ž
save/SaveV2/shape_and_slicesConst"/device:CPU:0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
­
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesSadanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_lossKadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/biasradanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weightadanet/iteration_0/step7adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias9adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel9adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias;adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernelSadanet/iteration_1/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_lossSadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/adanet_lossSadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/adanet_lossKadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/biasradanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weightradanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weightKadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/biasradanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weightradanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weightadanet/iteration_1/step7adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias9adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel9adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias;adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel7adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernelcurrent_iterationglobal_step"/device:CPU:0*,
dtypes"
 2				
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Ź
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*

axis *
N*
_output_shapes
:*
T0

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0

save/RestoreV2/tensor_namesConst"/device:CPU:0*˘
valueBBSadanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_lossBKadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/biasBradanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weightBadanet/iteration_0/stepB7adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/biasB9adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernelB9adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/biasB;adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernelBSadanet/iteration_1/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_lossBSadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/adanet_lossBSadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/adanet_lossBKadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/biasBradanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weightBradanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weightBKadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/biasBradanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weightBradanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weightBadanet/iteration_1/stepB7adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/biasB9adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernelB9adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/biasB;adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernelB7adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/biasB9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernelB9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/biasB;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernelB9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/biasB;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernelBcurrent_iterationBglobal_step*
dtype0*
_output_shapes
:
ą
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ą
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2				
Ź
save/AssignAssignSadanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_losssave/RestoreV2*
use_locking(*
T0*f
_class\
ZXloc:@adanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss*
validate_shape(*
_output_shapes
: 
¤
save/Assign_1AssignKadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/biassave/RestoreV2:1*
_output_shapes
:
*
use_locking(*
T0*^
_classT
RPloc:@adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/bias*
validate_shape(
ď
save/Assign_2Assignradanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weightsave/RestoreV2:2*
use_locking(*
T0*
_class{
ywloc:@adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight*
validate_shape(*
_output_shapes
: 
¸
save/Assign_3Assignadanet/iteration_0/stepsave/RestoreV2:3*
T0	**
_class 
loc:@adanet/iteration_0/step*
validate_shape(*
_output_shapes
: *
use_locking(
ü
save/Assign_4Assign7adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/biassave/RestoreV2:4*
use_locking(*
T0*J
_class@
><loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias*
validate_shape(*
_output_shapes
: 

save/Assign_5Assign9adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernelsave/RestoreV2:5*
use_locking(*
T0*L
_classB
@>loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel*
validate_shape(*
_output_shapes
:	 

save/Assign_6Assign9adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/biassave/RestoreV2:6*
use_locking(*
T0*L
_classB
@>loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias*
validate_shape(*
_output_shapes
:


save/Assign_7Assign;adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernelsave/RestoreV2:7*
T0*N
_classD
B@loc:@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel*
validate_shape(*
_output_shapes

: 
*
use_locking(
°
save/Assign_8AssignSadanet/iteration_1/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_losssave/RestoreV2:8*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*f
_class\
ZXloc:@adanet/iteration_1/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss
°
save/Assign_9AssignSadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/adanet_losssave/RestoreV2:9*
use_locking(*
T0*f
_class\
ZXloc:@adanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/adanet_loss*
validate_shape(*
_output_shapes
: 
˛
save/Assign_10AssignSadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/adanet_losssave/RestoreV2:10*
_output_shapes
: *
use_locking(*
T0*f
_class\
ZXloc:@adanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/adanet_loss*
validate_shape(
Ś
save/Assign_11AssignKadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/biassave/RestoreV2:11*^
_classT
RPloc:@adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/bias*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0
ń
save/Assign_12Assignradanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weightsave/RestoreV2:12*
use_locking(*
T0*
_class{
ywloc:@adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight*
validate_shape(*
_output_shapes
: 
ń
save/Assign_13Assignradanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weightsave/RestoreV2:13*
use_locking(*
T0*
_class{
ywloc:@adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight*
validate_shape(*
_output_shapes
: 
Ś
save/Assign_14AssignKadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/biassave/RestoreV2:14*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*^
_classT
RPloc:@adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/bias
ń
save/Assign_15Assignradanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weightsave/RestoreV2:15*
T0*
_class{
ywloc:@adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight*
validate_shape(*
_output_shapes
: *
use_locking(
ń
save/Assign_16Assignradanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weightsave/RestoreV2:16*
use_locking(*
T0*
_class{
ywloc:@adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight*
validate_shape(*
_output_shapes
: 
ş
save/Assign_17Assignadanet/iteration_1/stepsave/RestoreV2:17*
_output_shapes
: *
use_locking(*
T0	**
_class 
loc:@adanet/iteration_1/step*
validate_shape(
ţ
save/Assign_18Assign7adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/biassave/RestoreV2:18*
use_locking(*
T0*J
_class@
><loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias*
validate_shape(*
_output_shapes
: 

save/Assign_19Assign9adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernelsave/RestoreV2:19*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel*
validate_shape(*
_output_shapes
:	 *
use_locking(*
T0

save/Assign_20Assign9adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/biassave/RestoreV2:20*
T0*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias*
validate_shape(*
_output_shapes
:
*
use_locking(

save/Assign_21Assign;adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernelsave/RestoreV2:21*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel*
validate_shape(*
_output_shapes

: 
*
use_locking(*
T0
ţ
save/Assign_22Assign7adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/biassave/RestoreV2:22*
use_locking(*
T0*J
_class@
><loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias*
validate_shape(*
_output_shapes
: 

save/Assign_23Assign9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernelsave/RestoreV2:23*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel*
validate_shape(*
_output_shapes
:	 *
use_locking(*
T0

save/Assign_24Assign9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/biassave/RestoreV2:24*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0

save/Assign_25Assign;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernelsave/RestoreV2:25*
use_locking(*
T0*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel*
validate_shape(*
_output_shapes

:  

save/Assign_26Assign9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/biassave/RestoreV2:26*
use_locking(*
T0*L
_classB
@>loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias*
validate_shape(*
_output_shapes
:


save/Assign_27Assign;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernelsave/RestoreV2:27*
use_locking(*
T0*N
_classD
B@loc:@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel*
validate_shape(*
_output_shapes

: 

Ž
save/Assign_28Assigncurrent_iterationsave/RestoreV2:28*
T0	*$
_class
loc:@current_iteration*
validate_shape(*
_output_shapes
: *
use_locking(
˘
save/Assign_29Assignglobal_stepsave/RestoreV2:29*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@global_step

save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"ŻJ
	variablesĄJJ
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0

adanet/iteration_0/step:0adanet/iteration_0/step/Assignadanet/iteration_0/step/read:02+adanet/iteration_0/step/Initializer/zeros:0

;adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel:0@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/Assign@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/read:02Vadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/Initializer/random_uniform:08

9adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias:0>adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias/Assign>adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias/read:02Kadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias/Initializer/zeros:08
Ł
=adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel:0Badanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/AssignBadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/read:02Xadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/Initializer/random_uniform:08

;adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias:0@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias/Assign@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias/read:02Madanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias/Initializer/zeros:08
÷
tadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight:0yadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/Assignyadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/read:02adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/Initializer/Const:08
Ř
Madanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/bias:0Radanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/bias/AssignRadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/bias/read:02_adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/bias/Initializer/zeros:0
ô
Uadanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss:0Zadanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss/AssignZadanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss/read:02cadanet/iteration_0/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss/initial_value:0

adanet/iteration_1/step:0adanet/iteration_1/step/Assignadanet/iteration_1/step/read:02+adanet/iteration_1/step/Initializer/zeros:0
ô
Uadanet/iteration_1/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss:0Zadanet/iteration_1/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss/AssignZadanet/iteration_1/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss/read:02cadanet/iteration_1/candidate_t0_1_layer_dnn_grow_complexity_regularized/adanet_loss/initial_value:0

;adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel:0@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/Assign@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/read:02Vadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/Initializer/random_uniform:08

9adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias:0>adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias/Assign>adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias/read:02Kadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias/Initializer/zeros:08
Ł
=adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel:0Badanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/AssignBadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/read:02Xadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/Initializer/random_uniform:08

;adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias:0@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias/Assign@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias/read:02Madanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias/Initializer/zeros:08

;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel:0@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/Assign@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/read:02Vadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/Initializer/random_uniform:08

9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias:0>adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias/Assign>adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias/read:02Kadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias/Initializer/zeros:08
Ł
=adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel:0Badanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/AssignBadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/read:02Xadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/Initializer/random_uniform:08

;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias:0@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias/Assign@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias/read:02Madanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias/Initializer/zeros:08
Ł
=adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel:0Badanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/AssignBadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/read:02Xadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/Initializer/random_uniform:08

;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias:0@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias/Assign@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias/read:02Madanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias/Initializer/zeros:08
÷
tadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight:0yadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/Assignyadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/read:02adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/Initializer/Const:08
÷
tadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight:0yadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/Assignyadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/read:02adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/Initializer/Const:08
Ř
Madanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/bias:0Radanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/bias/AssignRadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/bias/read:02_adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/bias/Initializer/zeros:0
ô
Uadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/adanet_loss:0Zadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/adanet_loss/AssignZadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/adanet_loss/read:02cadanet/iteration_1/candidate_t1_1_layer_dnn_grow_complexity_regularized/adanet_loss/initial_value:0
÷
tadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight:0yadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/Assignyadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/read:02adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/Initializer/Const:08
÷
tadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight:0yadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/Assignyadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/read:02adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/Initializer/Const:08
Ř
Madanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/bias:0Radanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/bias/AssignRadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/bias/read:02_adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/bias/Initializer/zeros:0
ô
Uadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/adanet_loss:0Zadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/adanet_loss/AssignZadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/adanet_loss/read:02cadanet/iteration_1/candidate_t1_2_layer_dnn_grow_complexity_regularized/adanet_loss/initial_value:0
p
current_iteration:0current_iteration/Assigncurrent_iteration/read:02%current_iteration/Initializer/zeros:0"%
saved_model_main_op


group_deps"3
saved_model_train_op

adanet/iteration_1/NoOp"ę2
trainable_variablesŇ2Ď2

;adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel:0@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/Assign@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/read:02Vadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/kernel/Initializer/random_uniform:08

9adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias:0>adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias/Assign>adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias/read:02Kadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense/bias/Initializer/zeros:08
Ł
=adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel:0Badanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/AssignBadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/read:02Xadanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/kernel/Initializer/random_uniform:08

;adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias:0@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias/Assign@adanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias/read:02Madanet/iteration_0/subnetwork_t0_1_layer_dnn/dense_1/bias/Initializer/zeros:08
÷
tadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight:0yadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/Assignyadanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/read:02adanet/iteration_0/ensemble_t0_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/Initializer/Const:08

;adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel:0@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/Assign@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/read:02Vadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/kernel/Initializer/random_uniform:08

9adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias:0>adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias/Assign>adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias/read:02Kadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense/bias/Initializer/zeros:08
Ł
=adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel:0Badanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/AssignBadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/read:02Xadanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/kernel/Initializer/random_uniform:08

;adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias:0@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias/Assign@adanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias/read:02Madanet/iteration_1/subnetwork_t1_1_layer_dnn/dense_1/bias/Initializer/zeros:08

;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel:0@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/Assign@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/read:02Vadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/kernel/Initializer/random_uniform:08

9adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias:0>adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias/Assign>adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias/read:02Kadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense/bias/Initializer/zeros:08
Ł
=adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel:0Badanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/AssignBadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/read:02Xadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/kernel/Initializer/random_uniform:08

;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias:0@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias/Assign@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias/read:02Madanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_1/bias/Initializer/zeros:08
Ł
=adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel:0Badanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/AssignBadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/read:02Xadanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/kernel/Initializer/random_uniform:08

;adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias:0@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias/Assign@adanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias/read:02Madanet/iteration_1/subnetwork_t1_2_layer_dnn/dense_2/bias/Initializer/zeros:08
÷
tadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight:0yadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/Assignyadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/read:02adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/Initializer/Const:08
÷
tadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight:0yadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/Assignyadanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/read:02adanet/iteration_1/ensemble_t1_1_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/Initializer/Const:08
÷
tadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight:0yadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/Assignyadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/read:02adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_0/logits/mixture_weight/Initializer/Const:08
÷
tadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight:0yadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/Assignyadanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/read:02adanet/iteration_1/ensemble_t1_2_layer_dnn_grow_complexity_regularized/weighted_subnetwork_1/logits/mixture_weight/Initializer/Const:08"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0*Ć
predictş
3
images)

AKAKAKAK:0˙˙˙˙˙˙˙˙˙]
probabilitiesL
5adanet/iteration_1/best_predictions/strided_slice_3:0˙˙˙˙˙˙˙˙˙
W
	class_idsJ
3adanet/iteration_1/best_predictions/strided_slice:0	˙˙˙˙˙˙˙˙˙V
logitsL
5adanet/iteration_1/best_predictions/strided_slice_2:0˙˙˙˙˙˙˙˙˙
W
classesL
5adanet/iteration_1/best_predictions/strided_slice_1:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict