#ifndef POSE_H
#define POSE_H

#include "rn.h"
#include "quaternion.h"
#include "unit_complex.h"


namespace lietorch {


template<class Translation, class Rotation>
class TwistBase;
template<class Translation, class Rotation>
class PoseBase;

namespace internal {

template<class Translation, class Rotation>
struct traits<PoseBase<Translation, Rotation>>
{
	static constexpr int Dim = Translation::Dim + Rotation::Dim;
	static constexpr int ActDim = LIETORCH_POSITION_DIM;


	using Tangent = TwistBase<Translation, Rotation>;
	using Vector = torch::Tensor;
	using DataType = torch::Tensor;
};

template<class Translation, class Rotation>
struct traits<TwistBase<Translation, Rotation>>
{
	static constexpr int Dim = Translation::Tangent::Dim + Rotation::Tangent::Dim;

	using LieAlg = torch::Tensor;
	using LieGroup = PoseBase<Translation, Rotation>;
	using DataType = torch::Tensor;
};

}

template<class Translation, class Rotation>
class TwistBase : public Tangent<TwistBase<Translation,Rotation>>
{
	using Base = Tangent<TwistBase<Translation,Rotation>>;

	using LinearVelocity = typename Translation::Tangent;
	using AngularVelocity = typename Rotation::Tangent;

public:
	using Base::coeffs;
	LIETORCH_INHERIT_TANGENT_TRAITS

	TwistBase (const LinearVelocity &linear = LinearVelocity(), const AngularVelocity &angular = AngularVelocity());

	LieAlg generator(int i) const;
	LieAlg hat () const;
	LieGroup exp() const;
	TwistBase scale (const DataType &other) const;

	LinearVelocity linear () const;
	AngularVelocity angular () const;
};

template<class Translation, class Rotation>
class PoseBase : public LieGroup<PoseBase<Translation,Rotation>>
{
	using Base = LieGroup<PoseBase<Translation,Rotation>>;

public:
	using Base::coeffs;

	LIETORCH_INHERIT_GROUP_TRAITS


	PoseBase (const Translation &_position = Translation (), const Rotation &_orientation = Rotation ());
	PoseBase (const DataType &coeffs);

	PoseBase inverse () const;
	Tangent log () const;
	PoseBase compose (const PoseBase &other) const;
	DataType dist (const PoseBase &other, const DataType &weights) const;
	Vector act (const Vector &v) const;
	Tangent differentiate (const Vector &outerGradient,
					   const Vector &v,
					   const OpFcn &op = OpIdentity,
					   const boost::optional<torch::Tensor &> &jacobian = boost::none) const;

	Translation translation () const;
	Rotation rotation () const;
};

// Actual Definitions
using Position2 = Rn<2>;
using Position3 = Rn<3>;

using Velocity2 = VelocityRn<2>;
using Velocity3 = VelocityRn<3>;


using Pose3R4 = PoseBase<Position3, QuaternionR4>;
using Pose = PoseBase<Position3, Quaternion>;
using Twist = TwistBase<Velocity3, AngularVelocity>;

using Position = Position3;
using Velocity = Velocity3;

using Pose3R4 = PoseBase<Position3, QuaternionR4>;
using Pose = PoseBase<Position3, Quaternion>;
using Twist3R4 = TwistBase<Velocity3, QuaternionR4Velocity>;

using Pose2 = PoseBase<Position2, UnitComplex>;

using Position = Position3;
using Velocity = Velocity3;


template<class ToGroup>
ToGroup pose_cast (const Pose &pose);

template<>
inline Pose pose_cast<Pose> (const Pose &pose) {
	return pose;
}

template<>
inline Position pose_cast<Position> (const Pose &pose) {
	return pose.translation ();
}

template<>
inline Quaternion pose_cast<Quaternion> (const Pose &pose) {
	return pose.rotation ();
}





}





#endif // POSE_H
