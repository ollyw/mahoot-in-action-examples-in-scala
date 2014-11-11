import java.io.File
import org.apache.mahout.cf.taste.eval.{DataModelBuilder, RecommenderBuilder}
import org.apache.mahout.cf.taste.impl.common.FastByIDMap
import org.apache.mahout.cf.taste.impl.eval.{GenericRecommenderIRStatsEvaluator, AverageAbsoluteDifferenceRecommenderEvaluator}
import org.apache.mahout.cf.taste.impl.model.{GenericBooleanPrefDataModel, GenericDataModel}
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood
import org.apache.mahout.cf.taste.impl.recommender.{GenericBooleanPrefUserBasedRecommender, ItemUserAverageRecommender, GenericUserBasedRecommender}
import org.apache.mahout.cf.taste.impl.similarity.{LogLikelihoodSimilarity, PearsonCorrelationSimilarity}
import org.apache.mahout.cf.taste.model.{PreferenceArray, DataModel}
import org.apache.mahout.common.RandomUtils
import org.scalatest._
import scala.collection.JavaConversions._

class RecommenderSpec extends WordSpec with Matchers {
  val introModel = new FileDataModel(new File("intro.csv"))
  // Data from http://grouplens.org/node/73
  lazy val groupLensModel = new FileDataModel(new File("ua.base"))
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

      val score = evaluator.evaluate(userbasedRecommenderBuilder(), dataModelBuilder, groupLensModel, 0.9, 1.0)
      score should be (0.9 +- 0.05)
    }
  }

  // Listing 2.6
  // The SlopeOne recommender was deprecated in Mahoot 0.8
  // https://issues.apache.org/jira/browse/MAHOUT-1250
  "Item user average recommender" should {

    "have an error of 0.748 with Grouplens data" in {
      val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator

      val score = evaluator.evaluate(itemUserAverageRecommenderBuilder, dataModelBuilder, groupLensModel, 0.9, 1.0)
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
        evaluator.evaluate(userbasedRecommenderBuilder(10), booleanDataModelBuilder, groupLensModel, 0.9, 1.0)
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
        builder(10), booleanDataModelBuilder, groupLensModel, null, 10, GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0)

      // These figures are nothing like the book. 9% precision and recall seems too poor
      stats.getPrecision should be (0.09 +- 0.005)
      stats.getRecall should be (0.09 +- 0.005)
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
        builder(10), booleanDataModelBuilder, groupLensModel, null, 10, GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0)

      // These figures are nothing like the book. 9% precision and recall seems too poor
      stats.getPrecision should be (0.192 +- 0.005)
      stats.getRecall should be (0.226 +- 0.005) //
    }
  }

  def userbasedRecommenderBuilder(n: Int = 2) = new RecommenderBuilder() {
    def buildRecommender(dm: DataModel) = {
      val similarity = new PearsonCorrelationSimilarity(dm)
      val neighbourhood = new NearestNUserNeighborhood(n, similarity, dm)
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
}
