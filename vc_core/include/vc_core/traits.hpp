#ifndef VC_UTILS_TRAITS
#define VC_UTILS_TRAITS

namespace se
{
    namespace internal
    {
        /**
         * @brief Traits struct template.
         *
         * Specializations of this template are used by derived classes to define their traits and
         * by base classes to access those traits within their definitions in the CRTP. For more on
         * CRTP, refer to: https://en.cppreference.com/w/cpp/language/crtp.html
         * @tparam T Derived class type.
         */
        template <typename T>
        struct traits
        {
        };
    } // namespace internal
} // namespace se

#endif