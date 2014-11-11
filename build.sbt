name := """mahoot"""

version := "1.0"

scalaVersion := "2.11.4"

libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.1" % "test"

libraryDependencies += "org.apache.mahout" % "mahout-core" % "0.9"

libraryDependencies += "org.apache.mahout" % "mahout-examples" % "0.9"

scalacOptions in ThisBuild ++= Seq("-unchecked", "-deprecation")
