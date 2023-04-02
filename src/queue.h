#ifndef SEQUENTIALQUEUE_H
#define SEQUENTIALQUEUE_H

#include <stdint.h>
#include "stack.h"
struct Queue : public Stack
{
    int front, back;
};

#endif