import React from "react";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "./ui/button";

export default function ResourceCard({ title, content, url }) {
  return (
    <div className="p-4 max-w-sm sm:max-w-md md:max-w-xl mx-auto">
      <Card className="  shadow-lg rounded-xl transition-transform duration-300 hover:scale-105 h-68">
        <CardHeader>
          <CardTitle className="text-xl sm:text-2xl font-bold drop-shadow-md pt-2">
            {title}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className=" text-sm sm:text-base  leading-relaxed">{content}</p>
        </CardContent>
        <CardFooter className="flex justify-start pt-2">
          <a href={url}>
            <Button className="bg-white cursor-pointer text-cyan-600 hover:bg-blue-100 font-semibold px-4 pt-2 rounded-lg shadow-md transition-all text-sm sm:text-base">
              اضغط هنا
            </Button>
          </a>
        </CardFooter>
      </Card>
    </div>
  );
}
