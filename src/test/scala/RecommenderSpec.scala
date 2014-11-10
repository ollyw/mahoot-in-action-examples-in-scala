import java.io.File
import org.apache.mahout.cf.taste.eval.{DataModelBuilder, RecommenderBuilder}
import org.apache.mahout.cf.taste.impl.common.FastByIDMap
import org.apache.mahout.cf.taste.impl.eval.{GenericRecommenderIRStatsEvaluator, AverageAbsoluteDifferenceRecommenderEvaluator}
import org.apache.mahout.cf.taste.impl.model.GenericDataModel
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity
import org.apache.mahout.cf.taste.model.{PreferenceArray, DataModel}
import org.apache.mahout.common.RandomUtils
import org.scalatest._
import scala.collection.JavaConversions._

class RecommenderSpec extends WordSpec with Matchers {
  val model = new FileDataModel(new File("intro.csv"))
  RandomUtils.useTestSeed()

  "Recommender" should {

    "recommend item 104" in {
      val similarity = new PearsonCorrelationSimilarity(model)
      val neighbourhood = new NearestNUserNeighborhood(2, similarity, model)
      val recommender = new GenericUserBasedRecommender(model, neighbourhood, similarity)
      val recommendations = recommender.recommend(1, 1)
      recommendations.head.getItemID should be === 104
      recommendations.head.getValue should be === 4.257081f
    }

    "have an error of 1.0 using AverageAbsoluteDifferenceRecommender" in {
      val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator

      val score = evaluator.evaluate(recommenderBuilder, dataModelBuilder, model, 0.9, 1.0)
      score should be === 1.0
    }

    "have precision of 0.75 and recall of 1.0 using RecommenderIRStatsEvaluator" in {
      val evaluator = new GenericRecommenderIRStatsEvaluator()

      val stats = evaluator.evaluate(
         recommenderBuilder, dataModelBuilder, model, null, 2, GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0)

      stats.getPrecision should be === 0.75
      stats.getRecall should be === 1.0
    }

    def dataModelBuilder = new DataModelBuilder() {
      def buildDataModel(trainingData: FastByIDMap[PreferenceArray]) = new GenericDataModel(trainingData)
    }

    def recommenderBuilder = new RecommenderBuilder() {
      def buildRecommender(dm: DataModel) = {
        val similarity = new PearsonCorrelationSimilarity(dm)
        val neighbourhood = new NearestNUserNeighborhood(2, similarity, dm)
        new GenericUserBasedRecommender(dm, neighbourhood, similarity)
      }
    }
  }
}
