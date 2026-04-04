//Vars

vec2 CameraEyeAngles(float t) {
    float aniT = t*(1. - exp(-t*0.2))*ISUNRT - 0.3;
    return vec2(-0.05 - (-cos(aniT)*0.5 + 0.5)*0.5, mod(-2.2 + (-cos(aniT*1.1)*0.5 + 0.5)*2., 2.*PI));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec4 info = texture(iChannel0, fragCoord*IRES);
    if (iFrame == 0) {
        if (fragCoord.x < 10. && fragCoord.y < 1.) {
            if (fragCoord.x < 1.) info = vec4(0., 0., 0., 0.);
            else if (fragCoord.x < 2.) info = vec4(-0.05, -2.2, 0., 0.);
            else if (fragCoord.x < 3.) info = vec4(0., 0., 0., 1.);
            else if (fragCoord.x < 4.) info = vec4(30., 8., 33., 1.);
            else if (fragCoord.x < 5.) info = vec4(1.);
        }
    } else {
		if (fragCoord.x < 16. && fragCoord.y < 1.) {
            if (fragCoord.x < 1.) {
                //Mouse
                if (iMouse.z > 0.) {
                    if (info.w == 0.) {
                    	info.w = 1.;
                    	info.xy = iMouse.zw;
                    }
                } else info.w = 0.;
            } else if (fragCoord.x < 2.) {
                //Angles
                if (texture(iChannel0, vec2(4.5, 0.5)*IRES).x > 0.5) {
                    info.xy = CameraEyeAngles(iTime);
                    info.zw = info.xy;
                } else {
                    vec4 LMouse = texture(iChannel0, vec2(0.5, 0.5)*IRES);
                    if (LMouse.w == 0.)  info.zw = info.xy;
                    if (LMouse.w == 1.) {
                        info.x = info.z + (iMouse.y - LMouse.y)*0.01;
                        info.x = clamp(info.x, -2.8*0.5, 2.8*0.5);
                        //X led
                        info.y = info.w - (iMouse.x - LMouse.x)*0.02;
                        info.y = mod(info.y, 3.1415926*2.);
                    }
                }
            } else if (fragCoord.x < 3.) {
                //Player Eye
                vec2 Angles = texture(iChannel0, vec2(1.5, 0.5)*IRES).xy;
                if (texture(iChannel0, vec2(4.5, 0.5)*IRES).x > 0.5) Angles = CameraEyeAngles(iTime);
                info.xyz = normalize(vec3(cos(Angles.x)*sin(Angles.y), sin(Angles.x), cos(Angles.x)*cos(Angles.y)));
            } else if (fragCoord.x < 4.) {
                //Player Pos
                if (texture(iChannel0, vec2(4.5, 0.5)*IRES).x > 0.5) {
                    float aniT = iTime*(1. - exp(-iTime*0.2))*ISUNRT - 0.9;
                    info.xyz = vec3(28., 7.5, 40.) + vec3(0., (-cos(aniT*0.5)*0.5 + 0.5)*19., -(-cos(aniT)*0.5 + 0.5)*30.);
                } else {
                    float Speed = 5.*iTimeDelta;
                    if (texelFetch(iChannel1, ivec2(32, 0), 0).x > 0.) Speed = 16.*iTimeDelta;
                    vec3 Eye = texture(iChannel0, vec2(2.5, 0.5)*IRES).xyz;
                    if (texelFetch(iChannel1, ivec2(87, 0), 0).x > 0.) info.xyz += Eye*Speed; //W
                    if (texelFetch(iChannel1, ivec2(83, 0), 0).x > 0.) info.xyz -= Eye*Speed; //S
                    vec3 Tan = normalize(cross(vec3(Eye.x, 0., Eye.z), vec3(0., 1., 0.)));
                    if (texelFetch(iChannel1, ivec2(65, 0), 0).x > 0.) info.xyz -= Tan*Speed; //A
                    if (texelFetch(iChannel1, ivec2(68, 0), 0).x > 0.) info.xyz += Tan*Speed; //D
                }
            } else if (fragCoord.x < 5.) {
                //Animation flag
                if (iMouse.z > 0.) info.x = 0.;
            }
        }
    }
    fragColor = info;
}