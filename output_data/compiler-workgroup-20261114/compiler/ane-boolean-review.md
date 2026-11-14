# Boolean bitwise expression type

After the agenda scalar correction, the same requested-ANE case stopped before invocation at an integer bitwise AND. Its agenda entry combines two boolean masks. The canonical pointwise producer called _compute(bool,bool,bool), which returns I32 for arithmetic, and emitted an integer AND before converting back to bool.

_elementwise now uses BOOL specifically for bitwise_and/or/xor/not when the output and every actual tensor/scalar operand are boolean. Integer and mixed operand paths retain the existing arithmetic promotion. This uses the existing typed boolean primitive and Core ML logical operation; it does not add integer bitwise emulation or change numerical placement. It also reaches the existing boolean NOT case in _apply. AST/diff PASS and source construction records confirm BOOL AND/NOT and unchanged I32 AND (ane-boolean-source.json). No post-correction runtime result at this checkpoint.
