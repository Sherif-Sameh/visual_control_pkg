/*
 * Description:
 * Discrete low pass filter implementation using the manif library for Lie groups.
 * For more on manif, check out the GitHub page: https://github.com/artivis/manif
 */

#ifndef VC_SE_LOW_PASS
#define VC_SE_LOW_PASS

#include <utility>

#include <manif/manif.h>

namespace se
{
    /**
     * @brief Generic Low Pass Filter (LPF) template for arbitrary Lie Groups.
     *
     * This class template implements discrete first-order LPFs for Lie Groups, where simple
     * vector operations like addition and subtraction result in invalid group elements. The LPF
     * implements the following update formula: `x_k+1` = `x_k` + `coeff` * (`y_k+1` - `x_k`). For
     * Lie groups this translates to `x_k+1` = `x_k`.rplus(`coeff` * `y_k+1`.rminus(`x_k`)).
     * @tparam _Group Lie group, a derived class from `manif::LieGroupBase`.
     */
    template <class _Group>
    class LowPassFilter
    {
    public:
        using Scalar = typename manif::LieGroupBase<_Group>::Scalar;
        using State = typename manif::LieGroupBase<_Group>::LieGroup;
        using Tangent = typename manif::LieGroupBase<_Group>::Tangent;

    public:
        LowPassFilter(const Scalar coeff = 1, const State &x = State::Identity());

        auto getLPFCoeff() const -> Scalar;
        auto getState() const -> State;
        void getState(State &x) const;
        void setLPFCoeff(const Scalar coeff);
        /**
         * @brief Set the internal state of the LPF.
         *
         * @tparam StateArgs Types of input arguments.
         * @param[in] args Any set of arguments from which a `State` object can be constructed.
         */
        template <typename... StateArgs>
        void setState(StateArgs &&...args);

        /**
         * @brief Update the internal state of the LPF with the given inputs.
         *
         * @tparam StateArgs Types of input arguments.
         * @param[in] args Any set of arguments from which a `State` object can be constructed.
         */
        template <typename... StateArgs>
        void update(StateArgs &&...args);

    protected:
        Scalar m_coeff;
        State m_x;
    };

    // Definitions

    template <class _Group>
    LowPassFilter<_Group>::LowPassFilter(const Scalar coeff, const State &x)
        : m_coeff(coeff), m_x(x)
    {
    }

    template <class _Group>
    auto LowPassFilter<_Group>::getLPFCoeff() const -> Scalar
    {
        return m_coeff;
    }

    template <class _Group>
    auto LowPassFilter<_Group>::getState() const -> State
    {
        return m_x;
    }

    template <class _Group>
    void LowPassFilter<_Group>::getState(State &x) const
    {
        x = m_x;
    }

    template <class _Group>
    void LowPassFilter<_Group>::setLPFCoeff(const Scalar coeff)
    {
        m_coeff = coeff;
    }

    template <class _Group>
    template <typename... StateArgs>
    void LowPassFilter<_Group>::setState(StateArgs &&...args)
    {
        const State x_new(std::forward<StateArgs>(args)...);
        m_x = std::move(x_new);
    }

    template <class _Group>
    template <typename... StateArgs>
    void LowPassFilter<_Group>::update(StateArgs &&...args)
    {
        const State x_new(std::forward<StateArgs>(args)...);
        Tangent x_diff = x_new.minus(m_x);
        m_x = m_x.plus(m_coeff * x_diff);
    }
} // namespace se
#endif