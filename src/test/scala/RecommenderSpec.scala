import java.io.File
import org.apache.mahout.cf.taste.common.Weighting
import org.apache.mahout.cf.taste.eval.{DataModelBuilder, RecommenderBuilder}
import org.apache.mahout.cf.taste.impl.common.FastByIDMap
import org.apache.mahout.cf.taste.impl.eval.{GenericRecommenderIRStatsEvaluator, AverageAbsoluteDifferenceRecommenderEvaluator}
import org.apache.mahout.cf.taste.impl.model.{GenericBooleanPrefDataModel, GenericDataModel}
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel
import org.apache.mahout.cf.taste.impl.neighborhood.{ThresholdUserNeighborhood, NearestNUserNeighborhood}
import org.apache.mahout.cf.taste.impl.recommender.svd.{ALSWRFactorizer, SVDRecommender}
import org.apache.mahout.cf.taste.impl.recommender.{GenericBooleanPrefUserBasedRecommender, ItemUserAverageRecommender, GenericUserBasedRecommender}
import org.apache.mahout.cf.taste.impl.similarity._
import org.apache.mahout.cf.taste.model.{PreferenceArray, DataModel}
import org.apache.mahout.cf.taste.recommender.Recommender
import org.apache.mahout.cf.taste.similarity.precompute.example.GroupLensDataModel
import org.apache.mahout.common.RandomUtils
import org.scalatest._
import scala.collection.JavaConversions._

class RecommenderSpec extends WordSpec with Matchers {
  object SlowTest extends Tag("SlowTest")

  val introModel = new FileDataModel(new File("intro.csv"))
  // Data from http://grouplens.org/node/73
  lazy val groupLens10KModel = new FileDataModel(new File("ua.base"))
  lazy val groupLens10MModel = new GroupLensDataModel(new File("ratings.dat"))
  RandomUtils.useTestSeed()

  "User based recommender" should {

    // Listing 2.2
    "recommend item 104" in {
      val similarity = new PearsonCorrelationSimilarity(introModel)
      val neighbourhood = new NearestNUserNeighborhood(2, similarity, introModel)
      val recommender = new GenericUserBasedRecommender(introModel, neighbourhood, similarity)
      val recommendations = recommender.recommend(1, 1)
      recommendations.head.getItemID should be (104)
      recommendations.head.getValue should be (4.257081f)
    }

    // listing 2.3
    "have an error of 1.0 using AverageAbsoluteDifferenceRecommender" in {
      val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator

      val score = evaluator.evaluate(userbasedRecommenderBuilder(), dataModelBuilder, introModel, 0.9, 1.0)
      score should be (1.0)
    }

    // Listing 2.4
    "have precision of 0.75 and recall of 1.0 using RecommenderIRStatsEvaluator" in {
      val evaluator = new GenericRecommenderIRStatsEvaluator()

      val stats = evaluator.evaluate(
         userbasedRecommenderBuilder(), dataModelBuilder, introModel, null, 2, GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0)

      stats.getPrecision should be (0.75)
      stats.getRecall should be (1.0)
    }

    // Listing 2.5
    "have error of 0.9 with Grouplens data" in {
      val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator

      val score = evaluator.evaluate(userbasedRecommenderBuilder(), dataModelBuilder, groupLens10KModel, 0.9, 1.0)
      score should be (0.9 +- 0.05)
    }
  }

  // Listing 2.6
  // The SlopeOne recommender was deprecated in Mahoot 0.8
  // https://issues.apache.org/jira/browse/MAHOUT-1250
  "Item user average recommender" should {

    "have an error of 0.748 with Grouplens data" in {
      val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator

      val score = evaluator.evaluate(itemUserAverageRecommenderBuilder, dataModelBuilder, groupLens10KModel, 0.9, 1.0)
      score should be (0.78 +- 0.05)
    }

    def itemUserAverageRecommenderBuilder = new RecommenderBuilder() {
      def buildRecommender(dm: DataModel) = new ItemUserAverageRecommender(dm)
    }
  }

  // Listing 3.6
  "Recommender without preference values" should {
    "throw an exception when using Pearson Correlation Similarity" in {
      val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator

      an [IllegalArgumentException] should be thrownBy {
        evaluator.evaluate(userbasedRecommenderBuilder(10), booleanDataModelBuilder, groupLens10KModel, 0.9, 1.0)
      }
    }

    // Listing 3.7
    "have a precision and recall of 9% (book suggests it should be 24.7)" in {
      val evaluator = new GenericRecommenderIRStatsEvaluator()

      def builder(n: Int) = new RecommenderBuilder() {
        def buildRecommender(dm: DataModel) = {
          val similarity = new LogLikelihoodSimilarity(dm)
          val neighbourhood = new NearestNUserNeighborhood(n, similarity, dm)
          new GenericUserBasedRecommender(dm, neighbourhood, similarity)
        }
      }

      val stats = evaluator.evaluate(
        builder(10), booleanDataModelBuilder, groupLens10KModel, null, 10, GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0)

      // These figures are nothing like the book. 9% precision and recall seems too poor
      stats.getPrecision should be (0.09 +- 0.01)
      stats.getRecall should be (0.09 +- 0.01)
    }

    // Listing 3.7 with GenericBooleanPrefUserBasedRecommender
    "have a precision of 19.2% and recall of 22.6% (book suggests it should be 22.9)" in {
      val evaluator = new GenericRecommenderIRStatsEvaluator()

      def builder(n: Int) = new RecommenderBuilder() {
        def buildRecommender(dm: DataModel) = {
          val similarity = new LogLikelihoodSimilarity(dm)
          val neighbourhood = new NearestNUserNeighborhood(n, similarity, dm)
          new GenericBooleanPrefUserBasedRecommender(dm, neighbourhood, similarity)
        }
      }

      val stats = evaluator.evaluate(
        builder(10), booleanDataModelBuilder, groupLens10KModel, null, 10, GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0)

      // These figures are nothing like the book. 9% precision and recall seems too poor
      stats.getPrecision should be (0.192 +- 0.005)
      stats.getRecall should be (0.226 +- 0.005) //
    }

    // Listing 4.2 is likely to cause an out of memory according to the book, so let's not do that one

    // Listing 4.3
    "have error of 0.89 with 1M Grouplens with 100 neighbours and 5% of data used for evaluation" taggedAs(SlowTest) in {
      val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator

      val score = evaluator.evaluate(userbasedRecommenderBuilder(100), dataModelBuilder, groupLens10MModel, 0.95, 0.05)
      score should be (0.89 +- 0.05)
    }

    "have error of 0.85 using 1M Grouplens with 10 neighbours and 5% of data used for evaluation" taggedAs(SlowTest) in {
      val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator

      val score = evaluator.evaluate(userbasedRecommenderBuilder(10), dataModelBuilder, groupLens10MModel, 0.95, 0.05)
      score should be (0.85 +- 0.05) // The book says 10 neighbours returns a worse result, but with mahoot 0.9 it doesn't
    }

    "have error of 0.84using 1M Grouplens with ThresholdUserNeighbourhood of 0.7" taggedAs(SlowTest) in {
      val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator

      val builder = recommenderBuilder { dm =>
        val similarity = new PearsonCorrelationSimilarity(dm, Weighting.WEIGHTED)
        val neighbourhood = new ThresholdUserNeighborhood(0.7, similarity, dm)
        new GenericUserBasedRecommender(dm, neighbourhood, similarity)
      }

      evaluator.evaluate(builder, dataModelBuilder, groupLens10MModel, 0.95, 0.05) should be (0.84 +- 0.1)
    }

    "have error of 0.92 using 1M Grouplens with ThresholdUserNeighbourhood of 0.9" taggedAs(SlowTest) in {
      val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator

      val builder = recommenderBuilder { dm =>
        val similarity = new PearsonCorrelationSimilarity(dm, Weighting.WEIGHTED)
        val neighbourhood = new ThresholdUserNeighborhood(0.9, similarity, dm)
        new GenericUserBasedRecommender(dm, neighbourhood, similarity)
      }

      evaluator.evaluate(builder, dataModelBuilder, groupLens10MModel, 0.95, 0.05) should be (0.92 +- 0.1)
    }

    "have error of 0.77 using 1M Grouplens with weighted ThresholdUserNeighbourhood of 0.9" taggedAs(SlowTest) in {
      val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator

      val builder = recommenderBuilder { dm =>
        val similarity = new PearsonCorrelationSimilarity(dm, Weighting.WEIGHTED)
        val neighbourhood = new ThresholdUserNeighborhood(0.9, similarity, dm)
        new GenericUserBasedRecommender(dm, neighbourhood, similarity)
      }

      evaluator.evaluate(builder, dataModelBuilder, groupLens10MModel, 0.95, 0.05) should be (0.77 +- 0.05)
    }

    "have error of 0.82 using 1M Grouplens with weighted ThresholdUserNeighbourhood of 0.9" taggedAs(SlowTest) in {
      val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator

      val builder = recommenderBuilder { dm =>
        val similarity = new EuclideanDistanceSimilarity(dm)
        val neighbourhood = new ThresholdUserNeighborhood(0.9, similarity, dm)
        new GenericUserBasedRecommender(dm, neighbourhood, similarity)
      }

      evaluator.evaluate(builder, dataModelBuilder, groupLens10MModel, 0.95, 0.05) should be (0.82 +- 0.05)
    }

    // Listing 4.5 - VERY SLOW - TEA BREAK TIME
    "have error of 0.80 using 1M Grouplens with Spearman Correlation Similarity" taggedAs(SlowTest) in {
      val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator

      val builder = recommenderBuilder { dm =>
        val similarity = new SpearmanCorrelationSimilarity(dm)
        val neighbourhood = new ThresholdUserNeighborhood(0.9, similarity, dm)
        new GenericUserBasedRecommender(dm, neighbourhood, similarity)
      }

      evaluator.evaluate(builder, dataModelBuilder, groupLens10MModel, 0.95, 0.01) should be (0.80 +- 0.05)
    }

    // VERY SLOW - TEA BREAK TIME
    "have error of 0.68 using 1M Grouplens with Tanimoto Coefficient" taggedAs(SlowTest) in {
      val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator

      val builder = recommenderBuilder { dm =>
        val similarity = new TanimotoCoefficientSimilarity(dm)
        val neighbourhood = new ThresholdUserNeighborhood(0.70, similarity, dm)
        new GenericUserBasedRecommender(dm, neighbourhood, similarity)
      }

      // Not sure why the Tanimoto performs so badly compared to the suggested value of 0.82 in the book
      evaluator.evaluate(builder, dataModelBuilder, groupLens10MModel, 0.95, 0.03) should be (0.68 +- 0.1)
    }

    // VERY SLOW - TEA BREAK TIME
    "have error of 0.73 using 1M Grouplens with LogLikelihood" taggedAs(SlowTest) in {
      val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator

      val builder = recommenderBuilder { dm =>
        val similarity = new LogLikelihoodSimilarity(dm)
        val neighbourhood = new ThresholdUserNeighborhood(0.90, similarity, dm)
        new GenericUserBasedRecommender(dm, neighbourhood, similarity)
      }

      evaluator.evaluate(builder, dataModelBuilder, groupLens10MModel, 0.95, 0.03) should be (0.73 +- 0.02)
    }
  }

  "SVD Recommender" should {
    "have error of 0.69" in {
      val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator
      val builder = recommenderBuilder { dm => new SVDRecommender(dm, new ALSWRFactorizer(dm, 10, 0.05, 10)) }

      evaluator.evaluate(builder, dataModelBuilder, groupLens10MModel, 0.95, 0.05) should be (0.69 +- 0.01)
    }
  }

  def userbasedRecommenderBuilder(numNeighbours: Int = 2) = new RecommenderBuilder() {
    def buildRecommender(dm: DataModel) = {
      val similarity = new PearsonCorrelationSimilarity(dm)
      val neighbourhood = new NearestNUserNeighborhood(numNeighbours, similarity, dm)
      new GenericUserBasedRecommender(dm, neighbourhood, similarity)
    }
  }

  def dataModelBuilder = new DataModelBuilder() {
    def buildDataModel(trainingData: FastByIDMap[PreferenceArray]) = new GenericDataModel(trainingData)
  }

  def booleanDataModelBuilder = new DataModelBuilder() {
    def buildDataModel(trainingData: FastByIDMap[PreferenceArray]) = {
      new GenericBooleanPrefDataModel(GenericBooleanPrefDataModel.toDataMap(trainingData))
    }
  }

  // Java's interface based builders are so tedious. Create a functional wrapper
  def recommenderBuilder(buildingBlock: DataModel => Recommender) = new RecommenderBuilder() {
    def buildRecommender(dm: DataModel) = buildingBlock(dm)
  }
}
