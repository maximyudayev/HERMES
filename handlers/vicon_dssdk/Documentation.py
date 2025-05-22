################################################################################
# \mainpage Developers Manual
# 
# The Vicon DataStream Software Development Kit (SDK) allows easy programmable access 
# to the information contained in the Vicon DataStream. The function calls within 
# the SDK allow users to connect to and request data from the Vicon DataStream. 
# 
# To dive straight in look at the ViconDataStream class.
# 
# Conventions
# ===========
# 
# By default the global coordinate system is Z-Up, Y-Left.  
# This can be changed by using SetAxisMapping.
# 
# *Note:* From Blade 3.4.0 the default coordinate system of the SDK is Z-Up regardless of the (Z-Up/Y-Up) setting in Blade.  This allows the same client to work correctly with Blade, Tracker and so on.
# 
# Units
# -----
# 
# Positions are expressed in millimeters.  Rotation is expressed in radians.
# 
# Vectors and Matrices
# --------------------
# 
# Positions are passed as 3 elements corresponding to \f$(x,y,z)\f$
# 
# \f[
# \begin{pmatrix}
#   a_0 \\
#   a_1 \\
#   a_2 
# \end{pmatrix}
# \f]
# 
# 
# A 3\f$\times\f$3 matrix is passed row-wise as a vector of 9 elements:
# 
# \f[
# \begin{pmatrix}
#   a_0 & a_1 & a_2 \\
#   a_3 & a_4 & a_5 \\
#   a_6 & a_7 & a_8
# \end{pmatrix}
# \f]
# 
# Matrices are assumed to pre-multiply:
# 
# \f[
# \mathbf{A} \times \mathbf{B} \times \mathbf{C} = \mathbf{A} \times ( \mathbf{B} \times \mathbf{C} )
# \f]
# 
# 
# Euler Angles
# ------------
# 
# When used an XYZ euler angle \f$(x,y,z)\f$ is constructed:
# 
# \f[
# \mathbf{R}_x \times \mathbf{R}_y \times \mathbf{R}_z
# \f]
# 
# \f[
# \mathbf{R}_x \times ( \mathbf{R}_y \times \mathbf{R}_z )
# \f]
# 
# \f[
# \begin{pmatrix}
#   1 & 0 & 0 \\
#   0 & \cos x & -\sin x \\
#   0 & \cos x & \sin x
# \end{pmatrix}
# \begin{pmatrix}
#   \cos y & 0 & \sin y \\
#   0 & 1 & 0 \\
#   -\sin y & 0 & \cos y
# \end{pmatrix}
# \begin{pmatrix}
#   \cos z & -\sin z & 0 \\
#   \sin z & \cos z & 0 \\
#   0 & 0 & 1
# \end{pmatrix}
# \f]
# 
# \f[
# \begin{pmatrix}
#   \cos y \cos z & -\cos y \sin z & \sin y \\
#   \cos x \sin z + \sin x \sin y \cos z & \cos x \cos z - \sin x \sin y \sin z & -\sin x \cos y \\
#   \sin x \sin z - \cos x \sin y \cos z & \sin x \cos z + \cos x \sin y \sin z & \cos x \cos y
# \end{pmatrix}
# \f]
# 
# 
