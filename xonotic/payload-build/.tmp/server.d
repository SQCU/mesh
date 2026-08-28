test-server.dat: common/gamemodes/gamemode/payload/sv_payload.qc \
  lib/_all.inc lib/compiler.qh dpdefs/pre.qh server/sys-pre.qh \
  dpdefs/progsdefs.qh dpdefs/upstream/progsdefs.qc \
  dpdefs/dpextensions.qh dpdefs/upstream/dpextensions.qc \
  server/sys-post.qh dpdefs/post.qh lib/self.qh lib/macro.qh lib/bool.qh \
  lib/int.qh lib/warpzone/mathlib.qc lib/warpzone/mathlib.qh \
  lib/deglobalization.qh lib/accumulate.qh lib/float.qh lib/log.qh \
  lib/progname.qh lib/misc.qh lib/p99.qh lib/static.qh lib/vector.qh \
  lib/angle.qc lib/arraylist.qh lib/bits.qh lib/color.qh lib/string.qh \
  lib/nil.qh lib/sort.qh lib/oo.qh lib/counting.qh lib/i18n.qh \
  lib/map.qh lib/unsafe.qh lib/cvar.qh lib/defer.qh lib/draw.qh \
  lib/enumclass.qh lib/file.qh lib/functional.qh lib/intrusivelist.qh \
  lib/iter.qh lib/test.qh lib/json.qc lib/lazy.qh lib/linkedlist.qh \
  lib/markdown.qh lib/math.qh lib/net.qh lib/registry.qh lib/yenc.qh \
  lib/noise.qh lib/p2mathlib.qc lib/promise.qc lib/promise.qh \
  lib/random.qc lib/random.qh lib/registry_net.qh lib/replicate.qh \
  lib/sortlist.qc lib/sortlist.qh lib/spawnfunc.qh lib/stats.qh \
  lib/struct.qh lib/test.qc lib/urllib.qc lib/urllib.qh \
  lib/matrix/_mod.inc lib/matrix/command.qc lib/matrix/command.qh \
  lib/matrix/matrix.qh common/command/_mod.qh common/command/generic.qh \
  common/constants.qh common/command/markup.qh common/command/reg.qh \
  common/command/command.qh common/command/rpn.qh lib/matrix/matrix.qc \
  ecs/_mod.qh ecs/main.qh ecs/lib.qh ecs/components/_mod.qh \
  ecs/components/input.qh ecs/components/physics.qh ecs/events/_mod.qh \
  ecs/events/physics.qh ecs/systems/_mod.qh ecs/systems/input.qh \
  ecs/systems/physics.qh common/gamemodes/gamemode/payload/sv_payload.qh \
  common/gamemodes/gamemode/payload/payload.qh common/mapinfo.qh \
  common/util.qh common/teams.qh common/gamemodes/sv_rules.qh \
  common/mutators/base.qh common/mutators/events.qh \
  server/round_handler.qh server/scores.qh common/scores.qh \
  common/notifications/all.qh common/sounds/sound.qh \
  common/weapons/all.qh common/stats.qh server/client.qh server/utils.qh \
  server/intermission.qh common/replicate.qh common/sounds/all.qh \
  common/sounds/all.inc common/sounds/../teams.qh common/sounds/all.qc \
  server/compat/quake3.qh server/main.qh \
  common/mapobjects/teleporters.qh common/mapobjects/defs.qh \
  common/mapobjects/trigger/secret.qh \
  common/mutators/mutator/doublejump/doublejump.qh \
  common/mutators/mutator/itemstime/itemstime.qh \
  common/physics/player.qh common/state.qh \
  common/physics/movetypes/movetypes.qh common/weapons/config.qh \
  common/weapons/weapon.qh common/items/item/pickup.qh \
  common/items/item.qh common/models/all.qh common/models/model.qh \
  common/models/all.inc server/items/spawning.qh \
  common/resources/resources.qh common/resources/all.inc \
  common/effects/qc/_mod.qh common/effects/qc/casings.qh \
  common/effects/qc/damageeffects.qh common/effects/qc/gibs.qh \
  common/effects/qc/globalsound.qh server/chat.qh \
  common/effects/qc/lightningarc.qh common/effects/qc/modeleffects.qh \
  common/effects/qc/rubble.qh common/items/_mod.qh common/items/all.qh \
  common/items/item/_mod.qh common/items/item/ammo.qh \
  common/resources/sv_resources.qh server/items/items.qh \
  common/items/item/armor.qh common/items/item/health.qh \
  common/items/item/jetpack.qh common/mutators/mutator/powerups/_mod.qh \
  common/mutators/mutator/powerups/powerups.qh \
  common/mutators/mutator/status_effects/all.qh \
  common/mutators/mutator/powerups/sv_powerups.qh \
  common/mutators/mutator/powerups/powerup/_mod.qh \
  common/mutators/mutator/powerups/powerup/invisibility.qh \
  common/mutators/mutator/powerups/powerup/shield.qh \
  common/mutators/mutator/powerups/powerup/speed.qh \
  common/mutators/mutator/powerups/powerup/strength.qh \
  common/weapons/calculations.qh common/weapons/projectiles.qh \
  common/effects/all.qh common/effects/effect.qh common/effects/all.inc \
  common/effects/../teams.qh server/bot/api.qh common/weapons/_all.qh \
  common/weapons/all.inc common/weapons/weapon/blaster.qh \
  common/weapons/weapon/shotgun.qh common/weapons/weapon/machinegun.qh \
  common/weapons/weapon/mortar.qh common/weapons/weapon/minelayer.qh \
  common/weapons/weapon/electro.qh common/weapons/weapon/crylink.qh \
  common/weapons/weapon/vortex.qh common/weapons/weapon/hagar.qh \
  common/weapons/weapon/devastator.qh common/weapons/weapon/porto.qh \
  common/weapons/weapon/vaporizer.qh common/weapons/weapon/hook.qh \
  common/weapons/weapon/hlac.qh common/weapons/weapon/tuba.qh \
  common/weapons/weapon/rifle.qh common/weapons/weapon/fireball.qh \
  common/weapons/weapon/seeker.qh common/weapons/weapon/shockwave.qh \
  common/weapons/weapon/arc.qh common/notifications/all.inc \
  common/mapobjects/platforms.qh common/mapobjects/subs.qh \
  common/mapobjects/triggers.qh \
  common/mutators/mutator/waypoints/waypointsprites.qh \
  common/mutators/mutator/waypoints/all.qh \
  common/mutators/mutator/waypoints/all.inc server/mutators/_mod.qh \
  server/mutators/events.qh server/mutators/loader.qh \
  server/command/vote.qh server/damage.qh server/gamelog.qh \
  server/teamplay.qh server/world.qh
lib/_all.inc:
lib/compiler.qh:
dpdefs/pre.qh:
server/sys-pre.qh:
dpdefs/progsdefs.qh:
dpdefs/upstream/progsdefs.qc:
dpdefs/dpextensions.qh:
dpdefs/upstream/dpextensions.qc:
server/sys-post.qh:
dpdefs/post.qh:
lib/self.qh:
lib/macro.qh:
lib/bool.qh:
lib/int.qh:
lib/warpzone/mathlib.qc:
lib/warpzone/mathlib.qh:
lib/deglobalization.qh:
lib/accumulate.qh:
lib/float.qh:
lib/log.qh:
lib/progname.qh:
lib/misc.qh:
lib/p99.qh:
lib/static.qh:
lib/vector.qh:
lib/angle.qc:
lib/arraylist.qh:
lib/bits.qh:
lib/color.qh:
lib/string.qh:
lib/nil.qh:
lib/sort.qh:
lib/oo.qh:
lib/counting.qh:
lib/i18n.qh:
lib/map.qh:
lib/unsafe.qh:
lib/cvar.qh:
lib/defer.qh:
lib/draw.qh:
lib/enumclass.qh:
lib/file.qh:
lib/functional.qh:
lib/intrusivelist.qh:
lib/iter.qh:
lib/test.qh:
lib/json.qc:
lib/lazy.qh:
lib/linkedlist.qh:
lib/markdown.qh:
lib/math.qh:
lib/net.qh:
lib/registry.qh:
lib/yenc.qh:
lib/noise.qh:
lib/p2mathlib.qc:
lib/promise.qc:
lib/promise.qh:
lib/random.qc:
lib/random.qh:
lib/registry_net.qh:
lib/replicate.qh:
lib/sortlist.qc:
lib/sortlist.qh:
lib/spawnfunc.qh:
lib/stats.qh:
lib/struct.qh:
lib/test.qc:
lib/urllib.qc:
lib/urllib.qh:
lib/matrix/_mod.inc:
lib/matrix/command.qc:
lib/matrix/command.qh:
lib/matrix/matrix.qh:
common/command/_mod.qh:
common/command/generic.qh:
common/constants.qh:
common/command/markup.qh:
common/command/reg.qh:
common/command/command.qh:
common/command/rpn.qh:
lib/matrix/matrix.qc:
ecs/_mod.qh:
ecs/main.qh:
ecs/lib.qh:
ecs/components/_mod.qh:
ecs/components/input.qh:
ecs/components/physics.qh:
ecs/events/_mod.qh:
ecs/events/physics.qh:
ecs/systems/_mod.qh:
ecs/systems/input.qh:
ecs/systems/physics.qh:
common/gamemodes/gamemode/payload/sv_payload.qh:
common/gamemodes/gamemode/payload/payload.qh:
common/mapinfo.qh:
common/util.qh:
common/teams.qh:
common/gamemodes/sv_rules.qh:
common/mutators/base.qh:
common/mutators/events.qh:
server/round_handler.qh:
server/scores.qh:
common/scores.qh:
common/notifications/all.qh:
common/sounds/sound.qh:
common/weapons/all.qh:
common/stats.qh:
server/client.qh:
server/utils.qh:
server/intermission.qh:
common/replicate.qh:
common/sounds/all.qh:
common/sounds/all.inc:
common/sounds/../teams.qh:
common/sounds/all.qc:
server/compat/quake3.qh:
server/main.qh:
common/mapobjects/teleporters.qh:
common/mapobjects/defs.qh:
common/mapobjects/trigger/secret.qh:
common/mutators/mutator/doublejump/doublejump.qh:
common/mutators/mutator/itemstime/itemstime.qh:
common/physics/player.qh:
common/state.qh:
common/physics/movetypes/movetypes.qh:
common/weapons/config.qh:
common/weapons/weapon.qh:
common/items/item/pickup.qh:
common/items/item.qh:
common/models/all.qh:
common/models/model.qh:
common/models/all.inc:
server/items/spawning.qh:
common/resources/resources.qh:
common/resources/all.inc:
common/effects/qc/_mod.qh:
common/effects/qc/casings.qh:
common/effects/qc/damageeffects.qh:
common/effects/qc/gibs.qh:
common/effects/qc/globalsound.qh:
server/chat.qh:
common/effects/qc/lightningarc.qh:
common/effects/qc/modeleffects.qh:
common/effects/qc/rubble.qh:
common/items/_mod.qh:
common/items/all.qh:
common/items/item/_mod.qh:
common/items/item/ammo.qh:
common/resources/sv_resources.qh:
server/items/items.qh:
common/items/item/armor.qh:
common/items/item/health.qh:
common/items/item/jetpack.qh:
common/mutators/mutator/powerups/_mod.qh:
common/mutators/mutator/powerups/powerups.qh:
common/mutators/mutator/status_effects/all.qh:
common/mutators/mutator/powerups/sv_powerups.qh:
common/mutators/mutator/powerups/powerup/_mod.qh:
common/mutators/mutator/powerups/powerup/invisibility.qh:
common/mutators/mutator/powerups/powerup/shield.qh:
common/mutators/mutator/powerups/powerup/speed.qh:
common/mutators/mutator/powerups/powerup/strength.qh:
common/weapons/calculations.qh:
common/weapons/projectiles.qh:
common/effects/all.qh:
common/effects/effect.qh:
common/effects/all.inc:
common/effects/../teams.qh:
server/bot/api.qh:
common/weapons/_all.qh:
common/weapons/all.inc:
common/weapons/weapon/blaster.qh:
common/weapons/weapon/shotgun.qh:
common/weapons/weapon/machinegun.qh:
common/weapons/weapon/mortar.qh:
common/weapons/weapon/minelayer.qh:
common/weapons/weapon/electro.qh:
common/weapons/weapon/crylink.qh:
common/weapons/weapon/vortex.qh:
common/weapons/weapon/hagar.qh:
common/weapons/weapon/devastator.qh:
common/weapons/weapon/porto.qh:
common/weapons/weapon/vaporizer.qh:
common/weapons/weapon/hook.qh:
common/weapons/weapon/hlac.qh:
common/weapons/weapon/tuba.qh:
common/weapons/weapon/rifle.qh:
common/weapons/weapon/fireball.qh:
common/weapons/weapon/seeker.qh:
common/weapons/weapon/shockwave.qh:
common/weapons/weapon/arc.qh:
common/notifications/all.inc:
common/mapobjects/platforms.qh:
common/mapobjects/subs.qh:
common/mapobjects/triggers.qh:
common/mutators/mutator/waypoints/waypointsprites.qh:
common/mutators/mutator/waypoints/all.qh:
common/mutators/mutator/waypoints/all.inc:
server/mutators/_mod.qh:
server/mutators/events.qh:
server/mutators/loader.qh:
server/command/vote.qh:
server/damage.qh:
server/gamelog.qh:
server/teamplay.qh:
server/world.qh:
