#include "lietorch/pose.h"

using namespace lietorch;
using namespace torch;
using namespace std;

namespace lietorch {

// Instantiate implemented templates
template
class PoseBase<Position3, QuaternionR4>;
template
class TwistBase<Position3, QuaternionR4>;

template
class PoseBase<Position3, Quaternion>;
template
class TwistBase<Position3, Quaternion>;

template
class PoseBase<Position2, UnitComplex>;
template
class TwistBase<Position2, UnitComplex>;

namespace ops::pose {

static const Tensor eye3 = torch::eye (3);

Tensor expCoupling (const Tensor &theta) {
	Tensor skewRotation = ops::quaternion::skew (theta).squeeze ();
	Tensor angle = theta.norm ();

	if (angle.item().toFloat () < 1e-5)
		return eye3;
	else
		return eye3 + (1 - angle.cos ()) / (angle * angle) * skewRotation + (angle - angle.sin ()) / (angle.pow (3)) * skewRotation.mm (skewRotation);
}

Tensor expCouplingInverse (const Tensor &theta) {
	Tensor skewRotation = ops::quaternion::skew (theta).squeeze ();
	Tensor angle  = theta.norm ();

	if (angle.item().toFloat () < 1e-5)
		return eye3;
	else
		return eye3 - 0.5 * skewRotation + (1/(angle *angle) - (1 + angle.cos ())/(2 * angle * angle.sin ())) * skewRotation.mm (skewRotation);
}

/*
Tensor rightJacobianInv (const Tensor &q) {
	Tensor rotationLog = ops::quaternion::log (q);
	Tensor skewRotationLog = ops::quaternion::skew (rotationLog);
	return torch::eye (3, kFloat) + 0.5 * skewRotationLog + (1 / ());
}
*/
}

template<class Translation, class Rotation>
Translation PoseBase<Translation, Rotation>::translation () const {
	const int tDim = Translation::Dim;
	return Translation (coeffs.is_complex () ? real (coeffs.slice(0, 0, tDim)):
						   coeffs.slice(0,0,tDim));
}


template<class Translation, class Rotation>
Rotation PoseBase<Translation, Rotation>::rotation () const {
	const int rDim = Rotation::Dim;
	const int tDim = Translation::Dim;
	return Rotation (coeffs.slice(0, tDim, tDim + rDim));
}

template<class Translation, class Rotation>
PoseBase<Translation, Rotation>::PoseBase (const Translation &translation, const Rotation &rotation):
	Base(torch::cat({translation.coeffs, rotation.coeffs}))
{
}

template<class Translation, class Rotation>
PoseBase<Translation, Rotation>::PoseBase (const DataType &coeffs):
	Base(coeffs)
{}

template<class Translation, class Rotation>
PoseBase<Translation, Rotation> PoseBase<Translation, Rotation>::inverse () const
{
	Rotation inv = rotation().inverse();
	return PoseBase(inv * (translation().inverse()), inv);
}

template<>
typename PoseBase<Position3,Quaternion>::Tangent PoseBase<Position3,Quaternion>::log () const {
	Tensor rotationLog = rotation().log().coeffs;

	return Tangent (ops::pose::expCouplingInverse (rotationLog).matmul (translation().coeffs), rotationLog);
}

template<>
typename PoseBase<Position3, QuaternionR4>::Tangent PoseBase<Position3,QuaternionR4>::log () const {
	return Tangent (translation().log(), rotation().log());
}

// Composition is different according to each specialization
// Pose3R4
template<>
PoseBase<Position3, QuaternionR4> PoseBase<Position3, QuaternionR4>::compose (const PoseBase &other) const {
	return PoseBase (translation() * other.translation(), rotation() * other.rotation());
}

template<>
PoseBase<Position3, Quaternion> PoseBase<Position3, Quaternion>::compose (const PoseBase &other) const {
	return PoseBase (translation() * (rotation() * other.translation()), rotation() * other.rotation());
}

template<>
PoseBase<Position2, UnitComplex> PoseBase<Position2, UnitComplex>::compose (const PoseBase &other) const {
    return PoseBase (translation() * (rotation() * other.translation ()), rotation() * other.rotation ());
}

template<class Translation, class Rotation>
typename PoseBase<Translation,Rotation>::DataType
PoseBase<Translation,Rotation>::dist(const PoseBase &other, const DataType &weights) const {
	assert ((weights.dim() == 1 && weights.size(0) == 2) && "A pose must be weighted by a 1d vector of length 2");

	return translation().dist(other.translation(), weights[0].unsqueeze(0)) + rotation().dist(other.rotation(), weights[1].unsqueeze(0));
}

template<class Translation, class Rotation>
typename PoseBase<Translation, Rotation>::Vector
PoseBase<Translation, Rotation>:: PoseBase::act (const Vector &v) const {
	return rotation() * v + translation().coeffs;
}

template<>
typename PoseBase<Position3, Quaternion>::Tangent
PoseBase<Position3, Quaternion>::differentiate (const Vector &outerGradient, const Vector &v, const OpFcn &op, const boost::optional<torch::Tensor &> &jacobian) const {
	return Tangent (rotation().act (translation().differentiate (outerGradient, v, op).coeffs),
				 rotation().differentiate (outerGradient, v, op));
}

template<>
typename PoseBase<Position3, QuaternionR4>::Tangent
    PoseBase<Position3, QuaternionR4>::differentiate (const Vector &outerGradient, const Vector &v, const OpFcn &op, const boost::optional<torch::Tensor &> &jacobian) const {
	return Tangent (translation().differentiate (outerGradient, v, op).coeffs,
				 rotation().differentiate (outerGradient, v, op));
}

template<class Translation, class Rotation>
TwistBase<Translation, Rotation>::TwistBase(const LinearVelocity &linear, const AngularVelocity &angular):
	Base(torch::cat ({linear.coeffs, angular.coeffs}))
{}

template<class Translation, class Rotation>
typename TwistBase<Translation, Rotation>::LinearVelocity
TwistBase<Translation, Rotation>::linear () const {
	// + sign fixes ODR violation issue prior to C++ 17
	return coeffs.slice (0, 0, +LinearVelocity::Dim);
}

template<class Translation, class Rotation>
typename TwistBase<Translation, Rotation>::AngularVelocity
TwistBase<Translation, Rotation>::angular () const {
	// + sign fixes ODR violation issue prior to C++ 17
	return coeffs.slice (0, +LinearVelocity::Dim, +LinearVelocity::Dim + AngularVelocity::Dim);
}

template<>
typename TwistBase<Position3, Quaternion>::LieGroup
TwistBase<Position3, Quaternion>::exp () const {
	return LieGroup (ops::pose::expCoupling (angular().coeffs).matmul (linear().coeffs), angular().exp ().coeffs);
}

template<>
typename TwistBase<Position3, QuaternionR4>::LieGroup
    TwistBase<Position3, QuaternionR4>::exp () const
{
	return PoseBase<Position3, QuaternionR4> (linear().exp().coeffs, angular().exp().coeffs);
}

template<class Translation, class Rotation>
TwistBase<Translation, Rotation> TwistBase<Translation, Rotation>::scale(const TwistBase::DataType &other) const
{
	assert (other.sizes().size() == 1 && other.size(0) == 2 && "Scaling tensor must be 1D and with exactly two elemenents");

	return TwistBase (linear() * other[0], angular() * other[1]);
}

}
