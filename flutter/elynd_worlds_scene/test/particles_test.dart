// What drifts through the air (living.dart WorldParticles), painted for real.
import 'dart:ui' as ui;

import 'package:elynd_worlds_scene/elynd_worlds_scene.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('falling leaves cross the whole picture before they start again', (
    tester,
  ) async {
    const w = 300, h = 600;
    var low = 0;
    await tester.runAsync(() async {
      for (var t = 0.0; t < 60; t += 0.5) {
        final recorder = ui.PictureRecorder();
        WorldParticles(
          kind: 'leaves',
          clockSeconds: t,
        ).paint(Canvas(recorder), const Size(300, 600));
        final image = await recorder.endRecording().toImage(w, h);
        final bytes = (await image.toByteData(
          format: ui.ImageByteFormat.rawRgba,
        ))!;
        for (var y = h * 9 ~/ 10; y < h; y++) {
          for (var x = 0; x < w; x++) {
            if (bytes.getUint8((y * w + x) * 4 + 3) > 0) low++;
          }
        }
        image.dispose();
      }
    });
    // They once vanished at about 60% of the height and started again at the top.
    expect(low, greaterThan(0), reason: 'a leaf reaches the bottom tenth');
  });
}
