#pragma once

#include <stdexcept>
#include <string>

namespace sigmo_python
{

/**
 * @class SigmoPythonError
 * @brief Eccezione base per tutti gli errori del bridge Python-SIGMO.
 */
class SigmoPythonError : public std::runtime_error
{
public:
    explicit SigmoPythonError(const std::string &message)
        : std::runtime_error(message) {}
};

/**
 * @class InvalidGraphInputError
 * @brief Lanciata quando i dati CSR passati da Python sono malformati o incoerenti.
 * Esempio: row_offsets.size() != num_nodes + 1.
 */
class InvalidGraphInputError : public SigmoPythonError
{
public:
    explicit InvalidGraphInputError(const std::string &message)
        : SigmoPythonError("Invalid Input: " + message) {}
};

/**
 * @class InvalidScopeError
 * @brief Lanciata quando viene specificato uno scope non supportato (diverso da "data" o "query").
 */
class InvalidScopeError : public SigmoPythonError
{
public:
    explicit InvalidScopeError(const std::string &message)
        : SigmoPythonError("Invalid Scope: " + message) {}
};

/**
 * @class DeviceRuntimeError
 * @brief Lanciata quando un kernel SYCL fallisce o la GPU lancia un'eccezione asincrona.
 */
class DeviceRuntimeError : public SigmoPythonError
{
public:
    explicit DeviceRuntimeError(const std::string &message)
        : SigmoPythonError("GPU Kernel Error: " + message) {}
};

/**
 * @class OutOfDeviceMemoryError
 * @brief Lanciata quando l'allocazione USM sulla GPU fallisce.
 */
class OutOfDeviceMemoryError : public SigmoPythonError
{
public:
    explicit OutOfDeviceMemoryError(const std::string &message)
        : SigmoPythonError("GPU Memory Full: " + message) {}
};

} // namespace sigmo_python