/**
 * @file Types.hpp
 * @brief The types every layer of the mind agrees on.
 *
 * ChatMessage lives here, at the bottom of the stack, because three layers need it
 * and none of them may depend on the others: the freestanding forward pass produces
 * it, the agency loop assembles it, and a hosted runtime consumes it. Putting it in
 * any one of those would have forced the other two to depend upwards.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_INFER_TYPES_HPP
#    define LPL_INFER_TYPES_HPP

#    include <string>

namespace lpl::infer {

/// One turn in a conversation. `role` is "system", "user" or "assistant" — kept as
/// a string rather than an enum because it crosses into prompt templates verbatim,
/// and a mismatch there is a silent quality loss rather than a compile error.
struct ChatMessage {
    std::string role;
    std::string content;
};

} // namespace lpl::infer

#endif // LPL_INFER_TYPES_HPP
