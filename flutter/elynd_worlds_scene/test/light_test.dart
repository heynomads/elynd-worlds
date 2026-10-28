import 'package:elynd_worlds_scene/elynd_worlds_scene.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('each world gets the light of its time of day; a sleep world falls into night', () {
    expect(LightJourney.of(timeOfDay: 'dawn'), LightJourney.sunrise);
    expect(LightJourney.of(timeOfDay: 'Morning'), LightJourney.sunrise);
    expect(LightJourney.of(timeOfDay: 'golden hour'), LightJourney.deepen);
    expect(LightJourney.of(timeOfDay: 'dusk'), LightJourney.deepen);
    expect(LightJourney.of(timeOfDay: 'midday'), LightJourney.none);
    expect(LightJourney.of(timeOfDay: 'night'), LightJourney.none);
    expect(LightJourney.of(), LightJourney.none);
    expect(LightJourney.of(timeOfDay: 'morning', sleep: true), LightJourney.nightfall);
  });

  test('it starts exactly as painted and has come all the way at the end', () {
    expect(LightJourney.progress(0, 600), 0);
    expect(LightJourney.progress(300, 600), 0.5);
    expect(LightJourney.progress(600, 600), 1);
    expect(LightJourney.progress(700, 600), 1);
    expect(LightJourney.progress(10, 0), 0); // a still world never changes
    for (final light in LightJourney.values) {
      expect(light.matrix(0, opacity: 0.4), LightJourney.identity(opacity: 0.4), reason: '$light');
    }
    expect(LightJourney.none.matrix(1), LightJourney.identity());
  });

  test('a sunrise warms and brightens, nightfall dims and stays warm, and nothing passes white', () {
    // A grey pixel through the matrix: red, green and blue out, 0 to 1 (the offsets are 0 to 255).
    List<double> through(List<double> m, double grey) => [
      for (var row = 0; row < 3; row++)
        (m[row * 5] + m[row * 5 + 1] + m[row * 5 + 2]) * grey + m[row * 5 + 4] / 255,
    ];
    final sunrise = through(LightJourney.sunrise.matrix(1), 0.5);
    expect(sunrise[0], greaterThan(0.55)); // brighter
    expect(sunrise[0], greaterThan(sunrise[2])); // and warmer
    final night = through(LightJourney.nightfall.matrix(1), 0.5);
    expect(night[0], lessThan(0.45)); // darker
    expect(night[0], greaterThan(night[2])); // and warm, never bluer
    for (final light in LightJourney.values) {
      for (final p in [0.0, 0.5, 1.0]) {
        expect(through(light.matrix(p), 1).every((c) => c <= 1.0 + 1e-9), isTrue, reason: '$light at $p');
      }
    }
  });

  test('two coats paint exactly the light of the matrix, with no layer of their own', () {
    for (final light in LightJourney.values) {
      for (final p in [0.0, 0.3, 1.0]) {
        final m = light.matrix(p), c = light.coats(p);
        final add = [c.add.r, c.add.g, c.add.b];
        for (final grey in [0.0, 0.25, 0.8, 1.0]) {
          for (var row = 0; row < 3; row++) {
            // Through the matrix (its offsets 0 to 255), and through a black coat at `shade` with `add` summed in.
            final viaMatrix = m[row * 5 + row] * grey + m[row * 5 + 4] / 255;
            expect(grey * (1 - c.shade) + add[row], closeTo(viaMatrix, 1e-6), reason: '$light at $p');
          }
        }
      }
    }
  });
}
