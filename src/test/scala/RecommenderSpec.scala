import java.io.File
import org.apache.mahout.cf.taste.eval.{DataModelBuilder, RecommenderBuilder}
import org.apache.mahout.cf.taste.impl.common.FastByIDMap
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator
import org.apache.mahout.cf.taste.impl.model.GenericDataModel
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity
import org.apache.mahout.cf.taste.model.{PreferenceArray, DataModel}
import org.apache.mahout.common.RandomUtils
import org.scalatest._
import org.scalatest.matchers.ShouldMatchers
import scala.collection.JavaConversions._

class RecommenderSpec extends WordSpec with Matchers {
  "Recommender" should {

    "recommend item 104" in {
      val model = new FileDataModel(new File("intro.csv"))
      val similarity = new PearsonCorrelationSimilarity(model)
      val neighbourhood = new NearestNUserNeighborhood(2, similarity, model)
      val recommender = new GenericUserBasedRecommender(model, neighbourhood, similarity)
      val recommendations = recommender.recommend(1, 1)
      recommendations.head.getItemID should be === 104
      recommendations.head.getValue should be === 4.257081f
    }

    "have an error of 1.0" in {
      val model = new FileDataModel(new File("intro.csv"))
      RandomUtils.useTestSeed()
      val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator

      val builder = new RecommenderBuilder() {
        def buildRecommender(dm: DataModel) = {
          val similarity = new PearsonCorrelationSimilarity(dm)
          val neighbourhood = new NearestNUserNeighborhood(2, similarity, dm)
          new GenericUserBasedRecommender(dm, neighbourhood, similarity)
        }
      }

      val modelBuilder = new DataModelBuilder() {
        def buildDataModel(trainingData: FastByIDMap[PreferenceArray]) = new GenericDataModel(trainingData)
      }

      val score = evaluator.evaluate(builder, modelBuilder, model, 0.9, 1.0)
      score should be === 1.0
    }
  }
}
