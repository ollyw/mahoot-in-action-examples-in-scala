import java.io.File
import org.apache.mahout.cf.taste.eval.{DataModelBuilder, RecommenderBuilder}
import org.apache.mahout.cf.taste.impl.common.FastByIDMap
import org.apache.mahout.cf.taste.impl.eval.{GenericRecommenderIRStatsEvaluator, AverageAbsoluteDifferenceRecommenderEvaluator}
import org.apache.mahout.cf.taste.impl.model.GenericDataModel
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood
import org.apache.mahout.cf.taste.impl.recommender.{ItemUserAverageRecommender, GenericUserBasedRecommender}
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity
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

    "recommend item 104" in {
      val similarity = new PearsonCorrelationSimilarity(introModel)
      val neighbourhood = new NearestNUserNeighborhood(2, similarity, introModel)
      val recommender = new GenericUserBasedRecommender(introModel, neighbourhood, similarity)
      val recommendations = recommender.recommend(1, 1)
      recommendations.head.getItemID should be (104)
      recommendations.head.getValue should be (4.257081f)
    }

    "have an error of 1.0 using AverageAbsoluteDifferenceRecommender" in {
      val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator

      val score = evaluator.evaluate(userbasedRecommenderBuilder, dataModelBuilder, introModel, 0.9, 1.0)
      score should be (1.0)
    }

    "have precision of 0.75 and recall of 1.0 using RecommenderIRStatsEvaluator" in {
      val evaluator = new GenericRecommenderIRStatsEvaluator()

      val stats = evaluator.evaluate(
         userbasedRecommenderBuilder, dataModelBuilder, introModel, null, 2, GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0)

      stats.getPrecision should be (0.75)
      stats.getRecall should be (1.0)
    }

    "have error of 0.9 with Grouplens data" in {
      val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator

      val score = evaluator.evaluate(userbasedRecommenderBuilder, dataModelBuilder, groupLensModel, 0.9, 1.0)
      score should be (0.9 +- 0.05)
    }

    def userbasedRecommenderBuilder = new RecommenderBuilder() {
      def buildRecommender(dm: DataModel) = {
        val similarity = new PearsonCorrelationSimilarity(dm)
        val neighbourhood = new NearestNUserNeighborhood(2, similarity, dm)
        new GenericUserBasedRecommender(dm, neighbourhood, similarity)
      }
    }
  }

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

  def dataModelBuilder = new DataModelBuilder() {
    def buildDataModel(trainingData: FastByIDMap[PreferenceArray]) = new GenericDataModel(trainingData)
  }
}
