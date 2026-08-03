/**
 * @file AgentServer.hpp
 * @brief Exposing the tool surface over a standard agent protocol.
 *
 * So an external client can drive the same dispatcher the demon drives. The
 * protocol is a transport, not a dialect: local socket, agent protocol and ring
 * buffer must carry identical calls.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_BACKEND_AGENTSERVER_HPP
#    define LPL_LPL_BACKEND_AGENTSERVER_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::backend {

// TODO(lot 8): declarations only — no implementation yet.

} // namespace lpl::backend

#endif // LPL_LPL_BACKEND_AGENTSERVER_HPP
