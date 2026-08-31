#include "payload.qh"

float payload_unpack_push(int packed, int idx)
{
	return ((packed >> ((idx - 1) * PLC_PUSH_BITS)) & PLC_PUSH_MASK) / PLC_PUSH_SCALE;
}
