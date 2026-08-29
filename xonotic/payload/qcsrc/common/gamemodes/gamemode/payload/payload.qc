#include "payload.qh"

float payload_unpack_push(int packed, int idx)
{
	return ((packed >> ((idx - 1) * PLC_PUSH_BITS)) & PLC_PUSH_MASK) / PLC_PUSH_SCALE;
}

int payload_unpack_cart_state(int packed, int c)
{
	return (packed >> (c * PLC_CS_BITS)) & PLC_CS_MASK;
}
