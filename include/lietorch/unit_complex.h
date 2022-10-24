#ifndef UNIT_COMPLEX_H
#define UNIT_COMPLEX_H

#include "lietorch/liegroup.h"
#include "lietorch/tangent.h"
#include "rn.h"
#include <boost/none.hpp>

namespace lietorch {

class UnitComplex;
class ComplexVelocity;

namespace internal {

template<>
struct traits<ComplexVelocity>
{
	static constexpr int Dim = 1;

	using LieAlg = torch::Tensor;
	using LieGroup = UnitComplex;
};

template<>
struct traits<UnitComplex>
{
	static constexpr int Dim = 2;
	static constexpr int ActDim = 2;

	using Tangent = ComplexVelocity;
	using Vector = torch::Tensor;
	using DataType = torch::Tensor;
};

}


class ComplexVelocity : public Tangent<ComplexVelocity>
{
	using Base = Tangent<ComplexVelocity>;

public:
	LIETORCH_INHERIT_CONSTRUCTOR(ComplexVelocity)
	LIETORCH_INHERIT_TANGENT_TRAITS;

	using Base::coeffs;

	ComplexVelocity (float angle);
	LieAlg generator (int i) const;
	LieAlg hat () const;
	LieGroup exp () const;
	DataType norm () const;
	ComplexVelocity scale (const DataType &other) const;
};

class UnitComplex : public LieGroup<UnitComplex>
{
	using Base = LieGroup<UnitComplex>;

public:
	LIETORCH_INHERIT_CONSTRUCTOR(UnitComplex)
	LIETORCH_INHERIT_GROUP_TRAITS;

	using Base::coeffs;

	UnitComplex (float angle);

	UnitComplex inverse () const;
	ComplexVelocity log () const;
	UnitComplex compose (const UnitComplex &other) const;
	DataType dist (const UnitComplex &other, const DataType &weights) const;
	Vector act (const Vector &v) const;
	ComplexVelocity differentiate (const Vector &outerGradient,
								  const Vector &v,
								  const OpFcn &op = OpIdentity,
								  const boost::optional<torch::Tensor &> &jacobian = boost::none) const;
	Vector getJacobian (const Vector &v) const;

	torch::Tensor real () const;
	torch::Tensor imag () const;

	UnitComplex conj () const;

private:
	torch::Tensor complexCoeffs () const;
};
}

#endif // UNIT_COMPLEX_H



















