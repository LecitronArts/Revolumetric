void mainImage(out vec4 fragColor, in vec2 fragCoord) {
	vec4 Color=texture(iChannel0,fragCoord.xy/iResolution.xy);
    vec2 fragFloor=floor(fragCoord);
    if (iFrame==0) { //Initialization
        if (fragCoord.x<10. && fragCoord.y<1.) { //Store vars
            if (fragCoord.x<1.) Color=vec4(0.,0.,0.,0.); //Mouse
            else if (fragCoord.x<2.) Color=vec4(-0.5,-1.5,-0.5,-1.5); //Player Eye (Angles)
            else if (fragCoord.x<3.) Color=vec4(0.,0.,0.,1.); //Player Eye (Vector)
            else if (fragCoord.x<4.) Color=vec4(7.9,4.,3.9,1.); //Player Pos
            else if (fragCoord.x<5.) Color=vec4(0.7,0.,0.,1.); //Sun angles
            else if (fragCoord.x<6.) Color=vec4(0.,0.,0.,1.); //Sun vector
        }
    } else { //Update
		if (fragCoord.x<10. && fragCoord.y<1.) { //Update vars
            if (fragCoord.x<1.) { //Mouse
                if (iMouse.z>0.) {
                    if (Color.w==0.) {
                    	Color.w=1.;
                    	Color.xy=iMouse.zw;
                    }
                } else Color.w=0.;
            } else if (fragCoord.x<2.) { //Player Eye (Angles)
                vec4 LMouse=texture(iChannel0,vec2(0.5,0.5)*IRES);
                if (LMouse.w==0.)  Color.zw=Color.xy;
                if (LMouse.w==1.) {
                	Color.x=Color.z+(iMouse.y-LMouse.y)*0.01;
                	Color.x=clamp(Color.x,-2.8*0.5,2.8*0.5);
                	Color.y=Color.w-(iMouse.x-LMouse.x)*0.02;
               		Color.y=mod(Color.y,3.1415926*2.);
                }
            } else if (fragCoord.x<3.) { //Player Eye (Vector)
                vec3 Angles=texture(iChannel0,vec2(1.5,0.5)*IRES).xyz;
                Color.xyz=normalize(vec3(cos(Angles.x)*sin(Angles.y),
                  				sin(Angles.x),
                  				cos(Angles.x)*cos(Angles.y)));
            } else if (fragCoord.x<4.) { //Player Pos
                float Speed=iTimeDelta;
                	if (texelFetch(iChannel1,ivec2(32,0),0).x>0.) Speed=8.*iTimeDelta;
                vec3 Eye=texture(iChannel0,vec2(2.5,0.5)*IRES).xyz;
                if (texelFetch(iChannel1,ivec2(87,0),0).x>0.) Color.xyz+=Eye*Speed; //W
                if (texelFetch(iChannel1,ivec2(83,0),0).x>0.) Color.xyz-=Eye*Speed; //S
                vec3 Tan=normalize(cross(vec3(Eye.x,0.,Eye.z),vec3(0.,1.,0.)));
                if (texelFetch(iChannel1,ivec2(65,0),0).x>0.) Color.xyz-=Tan*Speed; //A
                if (texelFetch(iChannel1,ivec2(68,0),0).x>0.) Color.xyz+=Tan*Speed; //D
            } else if (fragCoord.x<5.) { //Sun angle
                if (texelFetch(iChannel1,ivec2(188,0),0).x>0.) Color.y+=0.02;
                if (texelFetch(iChannel1,ivec2(190,0),0).x>0.) Color.y-=0.02;
            } else if (fragCoord.x<6.) { //Sun direction
                vec2 Angles=texture(iChannel0,vec2(4.5,0.5)*IRES).xy;
                Color=vec4(normalize(vec3(cos(Angles.y)*cos(Angles.x)
                	,sin(Angles.x),sin(Angles.y)*cos(Angles.x))),1.);
            }
        }
    }
    fragColor=Color;
}
