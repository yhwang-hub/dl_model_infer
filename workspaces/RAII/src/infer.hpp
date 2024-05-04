#ifndef INFER_HPP
#define INFER_HPP

#include <memory>
#include <string>

class InferInterface
{
public:
    virtual void forward() = 0;
};

std::shared_ptr<InferInterface> create_infer(const std::string& file);

#endif //INFER_HPP