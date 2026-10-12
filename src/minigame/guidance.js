'use strict';
// Playground motion steering: only changes in image coordinates drive a turn.
// Profile values and output blend/slew limits come from the app; its absolute
// bearing guidance is intentionally not used for this motion-driven experiment.
class ProfileController {
  constructor(profile, camera, started) {
    this.profile=profile;this.camera=camera;this.started=started;
    this.previous=null;this.velocity=[0,0];this.command=[0,0,0];
    this.requested=[0,0,0];this.outputAt=started;this.stationary=true;
  }
  update(box,now) {
    const p=this.profile,previous=this.previous;
    this.previous=box?{box:box.slice(),t:now}:null;
    const sampleDt=previous?now-previous.t:0;
    const movement=box&&previous?sub(box,previous.box):[0,0];
    this.stationary=!box||!previous||sampleDt<=0||sampleDt>.25||norm(movement)<=1e-12;
    if(this.stationary){
      this.velocity=[0,0];this.requested=[0,0,0];this.command=[0,0,0];this.outputAt=now;
      return this.command;
    }
    // Normalize by simulation time, not render frames. Equal X/Y movement
    // produces the same request anywhere in the image; position adds no bias.
    const measured=mul(movement,1/sampleDt),oldVelocity=this.velocity;
    this.velocity=add(oldVelocity,mul(sub(measured,oldVelocity),sampleDt/(.08+sampleDt)));
    // Lead anticipates changes in image velocity. Constant motion has no
    // acceleration lead; stationary samples explicitly clear the motion estimate.
    const predicted=add(this.velocity,mul(sub(this.velocity,oldVelocity),p.lead/sampleDt));
    const x=predicted[0]*radians(this.camera.hfov),y=predicted[1]*radians(this.camera.vfov);
    const a=radians(p.camera_roll);
    // Turn opposite the image shift: a box moving left toward centre means
    // the drone turns right. Rotate image axes by the saved camera alignment.
    const rates=[-(x*Math.cos(a)-y*Math.sin(a))*p.gain,
                  (x*Math.sin(a)+y*Math.cos(a))*p.gain,0];
    this.requested=mul(rates,Math.min(1,radians(p.rate)/Math.max(norm(rates),1e-9)));
    const blend=smooth((now-this.started-p.straight)/p.blend);
    const delta=sub(mul(this.requested,blend),this.command);
    const dt=clamp(now-this.outputAt,0,.1);
    const slew=Math.min(1,2*radians(p.rate)/p.blend*dt/Math.max(norm(delta),1e-9));
    this.command=add(this.command,mul(delta,slew));this.outputAt=now;
    return this.command;
  }
}
