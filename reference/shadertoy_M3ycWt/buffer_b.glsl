//Volume (32 x 32 x 48)

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec4 info = texture(iChannel1, fragCoord*IRES);
    if (DFBox(fragCoord, vec2(256., 192.)) < 0.) {
        vec3 vPos = vec3(mod(fragCoord.x, 32.), floor(fragCoord.x*I32) + floor(fragCoord.y*I48)*8. + 0.5, mod(fragCoord.y, 48.));
        vec3 modP = vec3(vPos.xy, mod(vPos.z, 16.));
        info = vec4(0.);
        
        //Stone color
        vec3 stoneColor = vec3(0.95, 0.925, 0.9);
        
        //Bounding box
        if (DFBox(vPos - vec3(1., 1., 1.), vec3(30., 100., 46.)) > 0.) info = vec4(stoneColor, 1.);
        
        //Floor
        if (vPos.y < 2. && fract(dot(vPos.xz, vec2(0.707))*0.125) > 0.35 &&
            fract(dot(vPos.xz, vec2(0.707, -0.707))*0.125) > 0.35) info = vec4(stoneColor, 1.);
            //Temporal emissive
            if (info.w > 0.5 && abs(vPos.y - 1.5) < 0.5 &&
                length(vPos.xz - vec2(8., 8. + (sin(iTime - 1.5)*0.5 + 0.5)*24.)) < 5.)
                info = vec4(1.5, 0.7, 0.2, 2.);
        
        //Columns
        if (length(vec2(vPos.x - 16., mod(vPos.z + 8., 16.) - 8.)) < 2.) info = vec4(stoneColor, 1.);
            //Arcs
            if (vPos.y > 19.) modP.y -= 16.;
            if (DFBox(vec3(vPos.x, modP.y, vPos.z) - vec3(14., 9., 0.), vec3(3., 10., 32.)) < 0. &&
                length(modP.zy - vec2(8., 8.)) - abs(vPos.x - 15.5) > 6.) info = vec4(stoneColor, 1.);
            if (DFBox(vec3(vPos.x, modP.y, vPos.z) - vec3(16., 9., 30.), vec3(16., 10., 3.)) < 0. &&
                length(modP.xy - vec2(24., 8.)) - abs(vPos.z - 31.5) > 6.) info = vec4(stoneColor, 1.);
            //Cloth
            if (vPos.z < 32.) {
                vec3 aPos = vec3(abs(vPos.x - 18.), vPos.y, abs(vPos.z - 8.));
                if (DFBox(vPos - vec3(17., 3., 1.), vec3(2., 13., 14.)) < 0. &&
                    abs(vPos.x - (18.5 - float(vPos.y > 15. || aPos.z > 20. - vPos.y
                    || aPos.z < 2. + (17. - vPos.y)*(17. - vPos.y)*0.02))) < 0.25 &&
                    aPos.z > 1. + (17. - vPos.y)*(17. - vPos.y)*0.02) info = vec4(0.99, 0.4, 0.4, 1.);
                aPos = vec3(abs(vPos.x - 18.), vPos.y, abs(vPos.z - 24.));
                if (DFBox(vPos - vec3(17., 3., 17.), vec3(2., 13., 14.)) < 0. &&
                    abs(vPos.x - (18.5 - float(vPos.y > 15. || aPos.z > 20. - vPos.y
                    || aPos.z < 2. + (17. - vPos.y)*(17. - vPos.y)*0.02))) < 0.25 &&
                    aPos.z > 1. + (17. - vPos.y)*(17. - vPos.y)*0.02) info = vec4(0.4, 0.99, 0.4, 1.);
            }
        
        //Second floor
        if (DFBox(vPos - vec3(0., 15., 0.), vec3(16., 1., 48.)) < 0.) info = vec4(stoneColor, 1.);
        if (DFBox(vPos - vec3(0., 15., 32.), vec3(32., 1., 16.)) < 0.) info = vec4(stoneColor, 1.);
            //Inner arc
            if (vPos.x < 16. && mod(vPos.y, 16.) > 15. - pow(0.22*length(vec2(vPos.x - 8., modP.z - 8.)), 2.) &&
                length(modP - vec3(12., 5., 8.)) > 10.) info = vec4(stoneColor, 1.);
            if (vPos.x > 16. && vPos.z > 32. && mod(vPos.y, 16.) > 15. - pow(0.22*length(vec2(vPos.x - 24., modP.z - 8.)), 2.) &&
                length(modP - vec3(24., 5., 4.)) > 10.) info = vec4(stoneColor, 1.);
            //Fountain (?)
            if (vPos.y < 7. && length(vPos.xz - vec2(8., 40.)) < 2. + floor((vPos.y - 1.)*0.5) &&
                vPos.y < 3. + length(vPos.xz - vec2(8., 40.))) info = vec4(stoneColor, 1.);
                //Lamp above
                if (length(vec2(length(vPos.xz - vec2(8., 40.)) -
                    3.5 - floor((13.5 - vPos.y)*0.333), vPos.y - 13.5)) < 0.5) info = vec4(0.8, 0.6, 0.2, 2.);
                float lA = iTime;
        
        //X+ wall (bricks)
        if (vPos.x < 2. && mod(vPos.z + floor((vPos.y + 1.)/4.)*2., 4.) > 1. &&
            mod(vPos.y - 1., 4.) > 2.) info = vec4(stoneColor, 1.);
        
        //Z- wall (lion approx)
        if (smin(length(vPos - vec3(24., 8., 51.)) - 6.,
            length(vPos - vec3(24., 7., 45.5)) - 2., 4.) < 0. &&
            length(vec3(abs(vPos.x - 24.) - 2., vPos.y - 9., vPos.z - 44.)) > 1.) info = vec4(stoneColor, 1.);
        
        //Z+ wall
        if (vPos.x > 16.) {
            if (length(vec3(mod(vPos.x + 1., 8.) - 4., mod(vPos.y, 8.) - 4., vPos.z - 0.15)) < 2.5) info = vec4(stoneColor, 1.);
        }
        
        //Ceiling
        if (DFBox(vPos - vec3(0., 30., 0.), vec3(16., 2., 48.)) < 0.) info = vec4(stoneColor, 1.);
        if (DFBox(vPos - vec3(0., 30., 32.), vec3(32., 2., 16.)) < 0.) info = vec4(stoneColor, 1.);
        
        //Dynamic sphere
        if (length(vPos - vec3(8. + (sin(iTime - 0.3)*0.5 + 0.5)*19., 24., 8.)) < 4.) info = vec4(0.5, 0.6, 0.9, 1.);
    }
    fragColor = info;
}