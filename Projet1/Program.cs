// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
namespace MotifFinder
{
    using System;
    using System.Collections.Generic;
    using System.Linq; 
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Models;
    using Microsoft.ML.Probabilistic.Utilities;
    using Range = Microsoft.ML.Probabilistic.Models.Range;

    public class Program
    {
        // Define the True Motif globally for comparison across tests
        // Motif: A, C, [G/T], [Any], T, G, A, [A/C] - Length 8
        static DiscreteChar[] TrueMotifDist = new[]
        {
            NucleobaseDist(a: 0.8, c: 0.1, g: 0.05, t: 0.05), // A
            NucleobaseDist(a: 0.0, c: 0.9, g: 0.05, t: 0.05), // C
            NucleobaseDist(a: 0.0, c: 0.0, g: 0.5, t: 0.5),   // G or T
            NucleobaseDist(a: 0.25, c: 0.25, g: 0.25, t: 0.25), // Noise
            NucleobaseDist(a: 0.1, c: 0.1, g: 0.1, t: 0.7),   // T
            NucleobaseDist(a: 0.0, c: 0.0, g: 0.9, t: 0.1),   // G
            NucleobaseDist(a: 0.9, c: 0.05, g: 0.0, t: 0.05), // A
            NucleobaseDist(a: 0.5, c: 0.5, g: 0.0, t: 0.0),   // A or C
        };

        public static void Main()
        {
            Rand.Restart(1337);

            Console.WriteLine("==================================================");
            Console.WriteLine("   MOTIF FINDER - AUTOMATED STRESS TESTS");
            Console.WriteLine("==================================================");
            Console.WriteLine();

            // ---------------------------------------------------------
            // TEST 1: SIGNAL-TO-NOISE (Effect of Sequence Length)
            // ---------------------------------------------------------
            Console.WriteLine(">>> TEST 1: RATIO SIGNAL/BRUIT (Longueur L)");
            Console.WriteLine("    Taille d'Échantillon (N) fixée à 30");
            Console.WriteLine("--------------------------------------------------");

            int fixedN = 30;
            int[] testLengths = { 25, 100, 500, 1000 };

            foreach (var len in testLengths)
            {
                Console.Write($"L = {len,4} ... ");
                var score = RunExperiment(fixedN, len, out string consensus);
                PrintBar(score);
                Console.WriteLine($" Score: {score:0.00} | Consensus: {consensus}");
            }
            Console.WriteLine();

            // ---------------------------------------------------------
            // TEST 2: SAMPLE SIZE EFFICIENCY (Effect of N)
            // ---------------------------------------------------------
            Console.WriteLine(">>> TEST 2: EFFICACITÉ DES DONNÉES (Taille N)");
            Console.WriteLine("    Longueur de Séquence (L) fixée à 50");
            Console.WriteLine("--------------------------------------------------");

            int fixedL = 50;
            int[] testCounts = { 5, 10, 20, 50 };

            foreach (var count in testCounts)
            {
                Console.Write($"N = {count,4} ... ");
                var score = RunExperiment(count, fixedL, out string consensus);
                PrintBar(score);
                Console.WriteLine($" Score: {score:0.00} | Consensus: {consensus}");
            }

            Console.WriteLine("\nTerminé. Appuyez sur Entrée pour quitter.");
            Console.ReadKey();
        }

        /// <summary>
        /// Runs a single iteration of the Motif Finder model with specific parameters.
        /// </summary>
        /// <returns>A "Similarity Score" (0.0 to 1.0) indicating how close the inferred motif is to the true motif.</returns>
        private static double RunExperiment(int sequenceCount, int sequenceLength, out string inferredConsensus)
        {
            // 1. Sample Data
            var backgroundNucleobaseDist = NucleobaseDist(a: 0.25, c: 0.25, g: 0.25, t: 0.25);
            SampleMotifData(sequenceCount, sequenceLength, 0.8, TrueMotifDist, backgroundNucleobaseDist, out string[] sequenceData, out int[] _);

            // 2. Define Model
            int motifLength = TrueMotifDist.Length;
            
            // Priors
            Vector motifNucleobasePseudoCounts = PiecewiseVector.Constant(char.MaxValue + 1, 1e-6);
            motifNucleobasePseudoCounts['A'] = motifNucleobasePseudoCounts['C'] = motifNucleobasePseudoCounts['G'] = motifNucleobasePseudoCounts['T'] = 1.0;

            Range motifCharsRange = new Range(motifLength);
            VariableArray<Vector> motifNucleobaseProbs = Variable.Array<Vector>(motifCharsRange);
            motifNucleobaseProbs[motifCharsRange] = Variable.Dirichlet(motifNucleobasePseudoCounts).ForEach(motifCharsRange);

            var sequenceRange = new Range(sequenceCount);
            VariableArray<string> sequences = Variable.Array<string>(sequenceRange);

            // Latent variables
            VariableArray<int> motifPositions = Variable.Array<int>(sequenceRange);
            motifPositions[sequenceRange] = Variable.DiscreteUniform(sequenceLength - motifLength + 1).ForEach(sequenceRange);

            VariableArray<bool> motifPresence = Variable.Array<bool>(sequenceRange);
            motifPresence[sequenceRange] = Variable.Bernoulli(0.8).ForEach(sequenceRange); 

            // Generative Process
            using (Variable.ForEach(sequenceRange))
            {
                using (Variable.If(motifPresence[sequenceRange]))
                {
                    var motifChars = Variable.Array<char>(motifCharsRange);
                    motifChars[motifCharsRange] = Variable.Char(motifNucleobaseProbs[motifCharsRange]);
                    var motif = Variable.StringFromArray(motifChars);

                    var backgroundLengthRight = sequenceLength - motifLength - motifPositions[sequenceRange];
                    var backgroundLeft = Variable.StringOfLength(motifPositions[sequenceRange], backgroundNucleobaseDist);
                    var backgroundRight = Variable.StringOfLength(backgroundLengthRight, backgroundNucleobaseDist);

                    sequences[sequenceRange] = backgroundLeft + motif + backgroundRight;
                }

                using (Variable.IfNot(motifPresence[sequenceRange]))
                {
                    sequences[sequenceRange] = Variable.StringOfLength(sequenceLength, backgroundNucleobaseDist);
                }
            }

            // 3. Inference
            sequences.ObservedValue = sequenceData;
            var engine = new InferenceEngine();
            engine.ShowProgress = false; 
            engine.Compiler.RecommendedQuality = QualityBand.Experimental; 

            // Infer the PFM (Position Frequency Matrix)
            var posterior = engine.Infer<IList<Dirichlet>>(motifNucleobaseProbs);

            // 4. Evaluate Results
            // Calculate a score based on the probability mass assigned to the "True" dominant characters
            double totalScore = 0;
            string consensusStr = "";

            for (int i = 0; i < motifLength; i++)
            {
                var meanVector = posterior[i].GetMean();
                
                // Find the character with the highest probability in the TRUE distribution
                double maxTrueProb = 0;
                char maxTrueChar = '?';
                
                foreach(char c in new[] {'A', 'C', 'G', 'T'}) {
                    double p = TrueMotifDist[i][c];
                    if(p > maxTrueProb) { maxTrueChar = c; }
                }

                // Add the inferred probability of that character to the score
                totalScore += meanVector[maxTrueChar];

                // Build consensus string for display
                double maxInferred = 0;
                char bestInferred = '?';
                foreach(char c in new[] {'A', 'C', 'G', 'T'}) {
                    if(meanVector[c] > maxInferred) { maxInferred = meanVector[c]; bestInferred = c; }
                }
                consensusStr += bestInferred;
            }

            inferredConsensus = consensusStr;
            return totalScore / motifLength; // Normalize to 0-1
        }

        // --- Helpers (unchanged) ---

        private static void PrintBar(double score)
        {
            int width = 10;
            int filled = (int)(score * width);
            Console.ForegroundColor = score > 0.8 ? ConsoleColor.Green : (score > 0.5 ? ConsoleColor.Yellow : ConsoleColor.Red);
            Console.Write("[");
            Console.Write(new string('=', filled));
            Console.Write(new string(' ', width - filled));
            Console.Write("]");
            Console.ResetColor();
        }

        private static DiscreteChar NucleobaseDist(double a, double c, double g, double t)
        {
            Vector probs = PiecewiseVector.Zero(char.MaxValue + 1);
            probs['A'] = a;
            probs['C'] = c;
            probs['G'] = g;
            probs['T'] = t;
            return DiscreteChar.FromVector(probs);
        }

        private static void SampleMotifData(
            int sequenceCount,
            int sequenceLength,
            double motifPresenceProbability,
            DiscreteChar[] motif,
            DiscreteChar backgroundDist,
            out string[] sequenceData,
            out int[] motifPositionData)
        {
            sequenceData = new string[sequenceCount];
            motifPositionData = new int[sequenceCount];
            for (int i = 0; i < sequenceCount; ++i)
            {
                if (Rand.Double() <= motifPresenceProbability)
                {
                    motifPositionData[i] = Rand.Int(sequenceLength - motif.Length + 1);
                    var backgroundBeforeChars = Util.ArrayInit(motifPositionData[i], j => backgroundDist.Sample());
                    var backgroundAfterChars = Util.ArrayInit(sequenceLength - motif.Length - motifPositionData[i], j => backgroundDist.Sample());
                    var sampledMotifChars = Util.ArrayInit(motif.Length, j => motif[j].Sample());
                    sequenceData[i] = new string(backgroundBeforeChars) + new string(sampledMotifChars) + new string(backgroundAfterChars);
                }
                else
                {
                    motifPositionData[i] = -1;
                    var background = Util.ArrayInit(sequenceLength, j => backgroundDist.Sample());
                    sequenceData[i] = new string(background);
                }
            }
        }
    }
}