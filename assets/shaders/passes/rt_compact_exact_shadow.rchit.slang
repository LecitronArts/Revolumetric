/// CompactExact triangle closest-hit shader for shadow rays (direct lighting).
///
/// The direct-lighting shadow path only needs to mark the ray as occluded.
/// Triangle hits use BuiltInTriangleIntersectionAttributes rather than the
/// procedural RtSurfaceIntersectionAttributes, so a separate shader is required
/// even though the body is identical.

#include "rt_shadow_common.slang"

[shader("closesthit")]
void main(inout RtShadowPayload payload,
          in BuiltInTriangleIntersectionAttributes attr)
{
    payload.occluded = 1u;
}
